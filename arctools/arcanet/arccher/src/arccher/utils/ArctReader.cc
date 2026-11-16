// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* main.h                                                     (C) 2000-2026 */
/*                                                                           */
/* TODO.                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccher/utils/ArctReader.h"

#include "arccore/base/PlatformUtils.h"
#include "arccore/common/ArcaNetReader.h"
#include "arccore/common/CommandLineArguments.h"
#include "arccore/common/JSONReader.h"
#include "arccore/trace/ITraceMng.h"

#include <fstream>
#include <arccore/base/FatalErrorException.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccher
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArctReader::
ArctReader(ITraceMng* tm)
: TraceAccessor(tm)
, m_run(makeRef(new Arct()))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArctReader::
readFile(const String& file, JSONDocument& doc, JSONValue& root)
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

void ArctReader::
readCommonPart(RunParam2& runp, const JSONValue& arccher_part)
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
      info() << "Key : " << key << " -- Value : " << value.valueAsBool();
      runp.m_properties[key.localstr()] = (value.valueAsBool() ? "1" : "0");
    }

    // JSONDocument::ParseNumbersAsStringsFlag
    else if (value.isString()) {
      info() << "Key : " << key << " -- Value : " << value.valueAsStringView();
      runp.m_properties[key.localstr()] = value.valueAsStringView();
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArctReader::
firstReadArccherGeneralPart(const JSONValue& arccher_part)
{
  if (arccher_part.null())
    return;
  {
    const JSONValue node = arccher_part.child("run");
    if (!node.null()) {
      m_run->m_run.clear(); // TODO A voir dans les specs.
      for (auto elem : node.valueAsArray()) {
        m_run->m_run.add(elem.valueAsStringView());
      }
    }
  }
  readCommonPart(m_run->m_global_params, arccher_part);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArctReader::
firstReadArccherCodePart(const JSONValue& arccher_part, const String& code)
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

  m_run->m_specific_params[code.localstr()] = m_run->m_global_params;

  if (code_part.null())
    return;

  readCommonPart(m_run->m_specific_params[code.localstr()], code_part);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArctReader::
readCodesExecutors(std::unordered_map<std::string, CodeExecutor>& codes_executors, const JSONValue& code_or_exec_part, ArrayView<String> needed_executors)
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
              cmd_opt.m_optional = _JSONValueAsBool(node2);
            }
          }
          {
            const JSONValue node2 = elem.value().child("needed");
            if (!node2.null()) {
              cmd_opt.m_needed = _JSONValueAsBool(node2);
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArctReader::
readExecutors(std::unordered_map<std::string, CodeExecutor>& codes_executors, const JSONValue& arccher_part, ArrayView<String> needed_executors)
{
  const JSONValue code_or_exec_part = arccher_part.child("executors");
  readCodesExecutors(codes_executors, code_or_exec_part, needed_executors);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArctReader::
readCodes(std::unordered_map<std::string, CodeExecutor>& codes_executors, const JSONValue& arccher_part, ArrayView<String> needed_executors)
{
  const JSONValue code_or_exec_part = arccher_part.child("codes");
  readCodesExecutors(codes_executors, code_or_exec_part, needed_executors);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArctReader::
readJSON(CommandLineArguments& cla)
{
  String config_file = cla.getParameter("ConfigFile");
  String param_file = cla.getParameter("ParamFile");
  String acase = cla.getParameter("Case");

  m_run->m_global_params.m_properties["_ARCT_PATH"] = param_file;
  m_run->m_global_params.m_properties["_CASE_NAME"] = acase;

  info() << "ConfigFile : " << config_file;
  info() << "ParamFile : " << param_file;
  info() << "Case : " << acase;

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
  firstReadArccherGeneralPart(config_file_root.child("general").child("arccher"));
  for (JSONValue elem : json_values) {
    firstReadArccherGeneralPart(reader.extractProgPart(elem));
  }

  // On a le "run", on sait quelles parties lire.

  UniqueArray<String> codes;
  for (const String& elem : m_run->m_run) {
    if (!codes.contains(elem)) {
      codes.add(elem);
    }
  }

  for (const String& elem : codes) {
    firstReadArccherCodePart(config_file_root.child("general").child("arccher"), elem);
    for (JSONValue elem2 : json_values) {
      firstReadArccherCodePart(reader.extractProgPart(elem2), elem);
    }
  }

  UniqueArray<String> executors;
  for (const String& elem : codes) {
    String& executor = m_run->m_specific_params[elem.localstr()].m_properties["executor_name"];
    info() << "ccc : " << elem.localstr() << " " << executor;
    if (executor.empty()) {
      executor = "_NONE";
    }
    info() << "ddd : " << elem.localstr() << " " << executor;
    if (!executors.contains(executor)) {
      executors.add(executor);
    }
  }

  m_run->m_executors["_NONE"].m_cmdline = "{_CODE}";

  readExecutors(m_run->m_executors, config_file_root.child("general").child("arccher"), executors);
  readCodes(m_run->m_codes, config_file_root.child("general").child("arccher"), codes);
  for (JSONValue elem : json_values) {
    readExecutors(m_run->m_executors, reader.extractProgPart(elem), executors);
    readCodes(m_run->m_codes, reader.extractProgPart(elem), codes);
  }

  if (reader.computePath()) {
    info() << "ArcaNet full path: " << reader.path();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ArctReader::
_readAllFile(StringView filename, bool is_binary,
             Array<Byte>& out_bytes)
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

inline bool ArctReader::
_JSONValueAsBool(const JSONValue& value)
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arccher

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
