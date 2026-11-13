// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* main.cc                                                     (C) 2000-2026 */
/*                                                                           */
/* TODO.                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <arccore/base/FatalErrorException.h>
#include <arccore/base/PlatformUtils.h>
#include <arccore/base/Ref.h>
#include <arccore/base/String.h>

#include <arccore/common/CommandLineArguments.h>
#include <arccore/common/JSONReader.h>
#include <arccore/common/ArcaNetReader.h>
#include <arccore/common/ExceptionUtils.h>

#include <arccore/trace/ITraceMng.h>

#include <reproc++/reproc.hpp>
#include <reproc++/run.hpp>

#include <fstream>
#include <unordered_map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class RunException
: public std::exception
{
 public:

  explicit RunException(int code, const std::string& prog)
  : m_code(code)
  , m_prog(prog)
  {}

  int code() const noexcept { return m_code; }
  const std::string& prog() const noexcept { return m_prog; }

 private:

  int m_code = -1;
  std::string m_prog;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

struct CmdOpt
{
  String m_opt = "{}";
  String m_default = "";
  bool m_optional = true;
  bool m_needed = true;
};

struct CodeExecutor
{
  String m_cmdline;
  std::unordered_map<std::string, CmdOpt> m_cmdopt;
};

struct RunParam2
{
  std::unordered_map<std::string, String> m_properties;
  std::unordered_map<std::string, std::string> m_env_var;
};

struct Run
{
  UniqueArray<String> m_run;

  RunParam2 m_global_params;
  std::unordered_map<std::string, RunParam2> m_specific_params;

  std::unordered_map<std::string, CodeExecutor> m_codes;
  std::unordered_map<std::string, CodeExecutor> m_executors;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ByteType>
bool _readAllFile(StringView filename, bool is_binary,
                  Array<ByteType>& out_bytes)
{
  using namespace std;
  long unsigned int file_length = platform::getFileLength(filename);
  if (file_length == 0) {
    return true;
  }
  ifstream ifile;
  ios::openmode mode = ios::in;
  if (is_binary)
    mode |= ios::binary;
  ifile.open(filename.toStdStringView().data());
  if (ifile.fail()) {
    return true;
  }
  out_bytes.resize(file_length);
  ifile.read((char*)(out_bytes.data()), file_length);
  if (ifile.bad()) {
    return true;
  }
  // It is possible that the number of bytes read is less
  // than the file length, especially on Windows with text files
  // and carriage return conversion. Therefore, it is necessary to resize
  // the bytes to the correct length.
  size_t nb_read = ifile.gcount();
  out_bytes.resize(nb_read);
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void readCommonPart(RunParam2& runp, const JSONValue& arccher_part, ITraceMng* tm)
{
  for (auto elem : arccher_part.keyValueChildren()) {
    if (elem.value().isObject()) {
      continue;
    }
    String key = elem.name();
    if (key == "run") {
      continue;
    }
    JSONValue value = elem.value();
    if (value.null()) {
      continue;
    }
    if (key == "env_var") {
      for (const JSONKeyValue env_var_elem : value.keyValueChildren()) {
        String env_var_ = env_var_elem.name();
        String env_var_value = env_var_elem.value().valueAsStringView();
        runp.m_env_var[env_var_.localstr()] = env_var_value.localstr();
      }
    }

    else if (value.isBool()) {
      tm->info() << "Key : " << key << " -- Value : " << value.valueAsBool();
      runp.m_properties[key.localstr()] = (value.valueAsBool() ? "1" : "0");
    }

    // JSONDocument::ParseNumbersAsStringsFlag
    else if (value.isString()) {
      tm->info() << "Key : " << key << " -- Value : " << value.valueAsStringView();
      runp.m_properties[key.localstr()] = value.valueAsStringView();
    }
  }
}

void firstReadArccherGeneralPart(Run& runp, const JSONValue& arccher_part, ITraceMng* tm)
{
  if (arccher_part.null())
    return;
  {
    const JSONValue node = arccher_part.child("run");
    if (!node.null()) {
      runp.m_run.clear(); // TODO A voir dans les specs.
      for (auto elem : node.valueAsArray()) {
        runp.m_run.add(elem.valueAsStringView());
      }
    }
  }
  readCommonPart(runp.m_global_params, arccher_part, tm);
}

void firstReadArccherCodePart(Run& runp, const JSONValue& arccher_part, const String& code, ITraceMng* tm)
{
  JSONValue code_part = arccher_part.child(code);

  String code_used = code;

  if (code.contains("~")) {
    if (code_part.null()) {
      UniqueArray<String> splited;
      code.split(splited, '~');
      code_used = splited[0];
      code_part = arccher_part.child(code_used);
    }
  }

  runp.m_specific_params[code.localstr()] = runp.m_global_params;

  if (code_part.null())
    return;

  readCommonPart(runp.m_specific_params[code.localstr()], code_part, tm);
}

inline bool JSONValueAsBool(const JSONValue& value)
{
  if (value.isBool()) {
    return value.valueAsBool();
  }
  if (value.isString()) {
    StringView aaa = value.valueAsStringView();
    return (aaa == "true" || aaa == "True" || aaa == "1");
  }
  ARCCORE_FATAL("Value is not a boolean");
}

void readCodesExecutors(std::unordered_map<std::string, CodeExecutor>& codes_executors, const JSONValue& code_or_exec_part, ArrayView<String> needed_executors, ITraceMng* tm)
{
  if (code_or_exec_part.null())
    return;

  for (const auto& needed_exec : needed_executors) {

    JSONValue exec_part = code_or_exec_part.child(needed_exec);

    if (exec_part.null()) {
      String needed_exec_used = needed_exec;
      if (needed_exec_used.contains("~")) {
        UniqueArray<String> splited;
        needed_exec_used.split(splited, '~');
        needed_exec_used = splited[0];
        exec_part = code_or_exec_part.child(needed_exec_used);
      }
    }
    if (exec_part.null()) {
      ARCCORE_FATAL("{0} not found", needed_exec);
    }

    CodeExecutor exec;
    {
      const JSONValue node = exec_part.child("cmdline");
      if (!node.null()) {
        exec.m_cmdline = node.valueAsStringView();
      }
    }
    {
      const JSONValue node = exec_part.child("cmdopt");
      if (!node.null()) {
        for (auto elem : node.keyValueChildren()) {
          String key = elem.name();
          CmdOpt& cmd_opt = exec.m_cmdopt[key.localstr()];
          {
            const JSONValue node2 = elem.value().child("opt");
            if (!node2.null()) {
              cmd_opt.m_opt = node2.valueAsStringView();
            }
          }
          {
            const JSONValue node2 = elem.value().child("default");
            if (!node2.null()) {
              cmd_opt.m_default = node2.valueAsStringView();
            }
          }
          {
            const JSONValue node2 = elem.value().child("optional");
            if (!node2.null()) {
              cmd_opt.m_optional = JSONValueAsBool(node2);
            }
          }
          {
            const JSONValue node2 = elem.value().child("needed");
            if (!node2.null()) {
              cmd_opt.m_needed = JSONValueAsBool(node2);
            }
          }
          if (!cmd_opt.m_optional && !cmd_opt.m_needed) {
            ARCCORE_FATAL("Coherence error -- optional : {0} -- needed : {1} -- key : {2}", key, cmd_opt.m_optional, cmd_opt.m_needed);
          }
          if (cmd_opt.m_optional && cmd_opt.m_needed && cmd_opt.m_default.empty()) {
            ARCCORE_FATAL("Coherence error 2");
          }
        }
      }
    }

    codes_executors[needed_exec.localstr()] = exec;
  }
}

void readExecutors(std::unordered_map<std::string, CodeExecutor>& codes_executors, const JSONValue& arccher_part, ArrayView<String> needed_executors, ITraceMng* tm)
{
  const JSONValue code_or_exec_part = arccher_part.child("executors");
  readCodesExecutors(codes_executors, code_or_exec_part, needed_executors, tm);
}

void readCodes(std::unordered_map<std::string, CodeExecutor>& codes_executors, const JSONValue& arccher_part, ArrayView<String> needed_executors, ITraceMng* tm)
{
  const JSONValue code_or_exec_part = arccher_part.child("codes");
  readCodesExecutors(codes_executors, code_or_exec_part, needed_executors, tm);
}

void readFile(const String& file, JSONDocument& doc, JSONValue& root)
{
  if (file.empty())
    return;

  if (!platform::isFileReadable(file)) {
    ARCCORE_FATAL("File not found : {0}", file);
  }

  UniqueArray<Byte> bytes;

  if (_readAllFile(file, false, bytes)) {
    ARCCORE_FATAL("File not available");
  }

  doc.parse(bytes, file, (JSONDocument::ParseCommentsFlag | JSONDocument::ParseNumbersAsStringsFlag));
  root = doc.root();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void readJSON(Run& runp, CommandLineArguments& cla, ITraceMng* tm)
{
  String config_file = cla.getParameter("ConfigFile");
  String param_file = cla.getParameter("ParamFile");
  String acase = cla.getParameter("Case");

  runp.m_global_params.m_properties["_ARCT_PATH"] = param_file;
  runp.m_global_params.m_properties["_CASE_NAME"] = acase;

  tm->info() << "ConfigFile : " << config_file;
  tm->info() << "ParamFile : " << param_file;
  tm->info() << "Case : " << acase;

  if (param_file.empty()) {
    ARCCORE_FATAL("Param file must be specified");
  }

  JSONDocument config_file_doc;
  JSONDocument param_file_doc;

  JSONValue config_file_root;
  JSONValue param_file_root;

  readFile(config_file, config_file_doc, config_file_root);
  readFile(param_file, param_file_doc, param_file_root);

  ArcaNetReader reader(param_file_root);
  reader.init(acase, "arccher");
  reader.read();
  ArrayView<JSONValue> json_values = reader.jsonValues();

  // On construit le "run" et les valeurs par defaut.
  firstReadArccherGeneralPart(runp, config_file_root.child("general").child("arccher"), tm);
  for (JSONValue elem : json_values) {
    firstReadArccherGeneralPart(runp, reader.extractProgPart(elem), tm);
  }

  // On a le "run", on sait quelles parties lire.

  UniqueArray<String> codes;
  for (const String& elem : runp.m_run) {
    if (!codes.contains(elem)) {
      codes.add(elem);
    }
  }

  for (const String& elem : codes) {
    firstReadArccherCodePart(runp, config_file_root.child("general").child("arccher"), elem, tm);
    for (JSONValue elem2 : json_values) {
      firstReadArccherCodePart(runp, reader.extractProgPart(elem2), elem, tm);
    }
  }

  UniqueArray<String> executors;
  for (const String& elem : codes) {
    String& executor = runp.m_specific_params[elem.localstr()].m_properties["executor_name"];
    tm->info() << "ccc : " << elem.localstr() << " " << executor;
    if (executor.empty()) {
      executor = "_NONE";
    }
    tm->info() << "ddd : " << elem.localstr() << " " << executor;
    if (!executors.contains(executor)) {
      executors.add(executor);
    }
  }

  runp.m_executors["_NONE"].m_cmdline = "{_CODE}";

  readExecutors(runp.m_executors, config_file_root.child("general").child("arccher"), executors, tm);
  readCodes(runp.m_codes, config_file_root.child("general").child("arccher"), codes, tm);
  for (JSONValue elem : json_values) {
    readExecutors(runp.m_executors, reader.extractProgPart(elem), executors, tm);
    readCodes(runp.m_codes, reader.extractProgPart(elem), codes, tm);
  }

  if (reader.computePath()) {
    tm->info() << "ArcaNet full path: " << reader.path();
  }
}

String getVariable(RunParam2& code_params, CodeExecutor& code_executor, const String& variable_name, ITraceMng* tm)
{
  CmdOpt& cmd_opt = code_executor.m_cmdopt[variable_name.localstr()];

  String variable;
  if (code_params.m_properties.contains(variable_name.localstr())) {
    variable = code_params.m_properties[variable_name.localstr()];
  }
  else {
    if (!cmd_opt.m_optional) {
      ARCCORE_FATAL("{0} not optional", variable_name);
    }
    if (!cmd_opt.m_needed) {
      tm->info() << "Var name : " << variable_name << " -- Value : ";
      return "";
    }
    if (cmd_opt.m_default.empty()) {
      ARCCORE_FATAL("{0} need default", variable_name);
    }
    variable = cmd_opt.m_default;
  }

  tm->info() << "Var name : " << variable_name << " -- Value : " << variable;

  Int8 type_var = 0;
  StringBuilder new_cmd;
  Span<const Byte> cmd_prog = cmd_opt.m_opt.bytes();

  bool backslash = false;
  Integer pos_begin = -1;
  for (Integer i = 0; i < cmd_prog.size(); ++i) {

    if (!backslash) {
      if (cmd_prog[i] == '{') {
        if (pos_begin == -2) {
          ARCCORE_FATAL("Error already {}");
        }
        pos_begin = i;
      }

      else if (cmd_prog[i] == '}') {
        if (pos_begin == -1) {
          ARCCORE_FATAL("Error {");
        }
        if (pos_begin + 1 != i) {
          ARCCORE_FATAL("Error not {}");
        }
        if (type_var == 1) {
          ARCCORE_FATAL("Error multiple {} not supported");
        }
        new_cmd += variable;
        pos_begin = -2;
        type_var = 1;
      }

      else if (pos_begin != -1) {
        ARCCORE_FATAL("Error not {}");
      }

      else {
        new_cmd += static_cast<char>(cmd_prog[i]);
      }
    }

    // Fin
    if (cmd_prog[i] == '\\') {
      backslash = true;
    }
  }

  if (pos_begin >= 0) {
    ARCCORE_FATAL("Error } -- Variable name : {0}", variable_name);
  }

  if (type_var == 0 && variable != "1") {
    tm->info() << "Var name : " << variable_name << " -- Final value : ";
    return "";
  }
  tm->info() << "Var name : " << variable_name << " -- Final value : " << new_cmd;
  return new_cmd;
}

String replaceVariables(RunParam2& code_params, CodeExecutor& code_executor, const String& cmdline, ITraceMng* tm)
{
  tm->info() << "cmdline : " << cmdline;
  StringBuilder new_cmd;
  Span<const Byte> cmd_prog = cmdline.bytes();

  bool backslash = false;
  Integer pos_begin = -1;
  UniqueArray<Byte> var_build;
  for (Integer i = 0; i < cmd_prog.size(); ++i) {

    if (!backslash) {
      if (cmd_prog[i] == '{') {
        pos_begin = i;
      }

      else if (cmd_prog[i] == '}') {
        if (pos_begin == -1) {
          ARCCORE_FATAL("Error {");
        }
        new_cmd += getVariable(code_params, code_executor, Span<const Byte>(var_build), tm);
        var_build.clear();
        pos_begin = -1;
      }

      else if (pos_begin != -1) {
        var_build.add(cmd_prog[i]);
      }

      else {
        new_cmd += static_cast<char>(cmd_prog[i]);
      }
    }

    // Fin
    if (cmd_prog[i] == '\\') {
      backslash = true;
    }
  }

  if (pos_begin != -1) {
    ARCCORE_FATAL("Error }");
  }
  return new_cmd;
}

String buildCmdLine(Run& runp, const String& prog, ITraceMng* tm)
{
  std::string prog_str = prog.localstr();

  CodeExecutor& code = runp.m_codes[prog_str];
  RunParam2& code_params = runp.m_specific_params[prog_str];

  CodeExecutor& exec = runp.m_executors[code_params.m_properties["executor_name"].localstr()];

  tm->info() << "executor_name : " << code_params.m_properties["executor_name"];
  String cmd_prog = code.m_cmdline;
  String cmd_exec = exec.m_cmdline;

  auto aaa = replaceVariables(code_params, code, cmd_prog, tm);
  tm->info() << "aaa : " << aaa;

  code_params.m_properties["_CODE"] = aaa;

  auto bbb = replaceVariables(code_params, exec, cmd_exec, tm);
  tm->info() << "bbb : " << bbb;

  return bbb;
}

void cmdLineToArray(const String& cmdline, UniqueArray<std::string>& cmdlinearray)
{
  Span<const Byte> cmd_prog = cmdline.bytes();

  Integer pos_begin = 0;
  bool previous_space = true;
  for (Integer i = 0; i < cmd_prog.size(); ++i) {
    if (cmd_prog[i] == ' ' && previous_space) {
      pos_begin++;
    }
    else if (cmd_prog[i] == ' ') {
      String aaa = cmd_prog.subSpan(pos_begin, i - pos_begin);
      cmdlinearray.add(aaa.localstr());
      pos_begin = i + 1;
      previous_space = true;
    }
    else {
      previous_space = false;
    }
  }
  if (pos_begin != cmd_prog.size()) {
    String aaa = cmd_prog.subSpan(pos_begin, cmd_prog.size() - pos_begin);
    cmdlinearray.add(aaa.localstr());
  }
}

void runCmd(Run& runp, const String& prog, const String& cmdline, ITraceMng* tm)
{
  std::string prog_local = prog.localstr();

  reproc::process process;
  reproc::options options;

  options.redirect.out.type = reproc::redirect::parent;
  options.redirect.err.type = reproc::redirect::parent;

  options.env.behavior = reproc::env::type::extend;

  if (!runp.m_specific_params[prog_local].m_env_var.empty())
    options.env.extra = runp.m_specific_params[prog_local].m_env_var;

  if (!runp.m_specific_params[prog_local].m_properties["working_dir"].empty())
    options.working_directory = runp.m_specific_params[prog_local].m_properties["working_dir"].localstr();

  UniqueArray<std::string> cmd;
  cmdLineToArray(cmdline, cmd);

  tm->info() << "cmd : " << cmd;

  {
    std::error_code ec = process.start(cmd, options);
    if (ec) {
      throw std::system_error(ec);
    }
  }

  // auto [status, ec] = reproc::run(cmd, options);

  auto [status, ec] = process.wait(reproc::infinite);
  if (ec) {
    throw std::system_error(ec);
  }
  if (status) {
    throw RunException(status, prog_local);
  }
}

void run(Run& runp, ITraceMng* tm)
{
  for (const auto& prog : runp.m_run) {
    tm->info() << "prog : " << prog;
    String cmd = buildCmdLine(runp, prog, tm);
    runCmd(runp, prog, cmd, tm);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int main(int argc, char* argv[])
{
  Ref<ITraceMng> tm_ref;
  ITraceMng* tm = nullptr;

  try {
    tm_ref = makeRef(arccoreCreateDefaultTraceMng());
    tm = static_cast<ITraceMng*>(ARCCORE_CHECK_POINTER(tm_ref.get()));
    tm->pushTraceClass("ArcCher");
  }
  catch (const std::exception& ex) {
    return ExceptionUtils::print(ex, nullptr, false);
  }
  catch (...) {
    return ExceptionUtils::print(nullptr, false);
  }

  try {
    CommandLineArguments cla(&argc, &argv);
    Run runp;
    readJSON(runp, cla, tm);
    run(runp, tm);
  }
  catch (const RunException& e) {
    std::cout << std::endl;
    tm->info() << "Can't continue with the execution.";
    tm->info() << "Program: " << e.prog();
    tm->info() << "Process return: " << e.code();
    return e.code();
  }
  catch (const std::system_error& e) {
    std::cout << std::endl;
    tm->info() << "Can't continue with the execution.";
    tm->info() << "Process return: " << e.code().value();
    tm->info() << "Process message: " << e.code().message();
    tm->info() << "Process what: " << e.what();
    return e.code().value();
  }
  catch (const std::exception& ex) {
    return ExceptionUtils::print(ex, tm, false);
  }
  catch (...) {
    return ExceptionUtils::print(tm, false);
  }

  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
