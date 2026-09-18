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

struct RunParam
{
  String m_param_file;
  String m_case;
  String m_app_exec_path;
  String m_mpi_exec_path;
  String m_working_dir;
  Integer m_mpi_nb_proc = 0;
  std::unordered_map<std::string, std::string> m_env_var;
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

void readArccherPart(RunParam& runp, const JSONValue& arccher_part)
{
  if (arccher_part.null())
    return;
  {
    const JSONValue node = arccher_part.child("code_binary");
    if (!node.null()) {
      runp.m_app_exec_path = node.valueAsStringView();
    }
  }
  {
    const JSONValue node = arccher_part.child("mpi_binary");
    if (!node.null()) {
      runp.m_mpi_exec_path = node.valueAsStringView();
    }
  }
  {
    const JSONValue node = arccher_part.child("mpi");
    if (!node.null()) {
      runp.m_mpi_nb_proc = std::stoi(String(node.valueAsStringView()).localstr());
    }
  }
  {
    const JSONValue node = arccher_part.child("working_dir");
    if (!node.null()) {
      runp.m_working_dir = node.valueAsStringView();
    }
  }
  {
    const JSONValue node = arccher_part.child("env_var");
    if (!node.null()) {
      for (const JSONKeyValue elem : node.keyValueChildren()) {
        String key = elem.name();
        String value = elem.value().valueAsStringView();
        runp.m_env_var[key.localstr()] = value.localstr();
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void readJSON(RunParam& runp, CommandLineArguments& cla, ITraceMng* tm)
{

  String config_file = cla.getParameter("ConfigFile");
  String param_file = cla.getParameter("ParamFile");
  String acase = cla.getParameter("Case");

  runp.m_param_file = param_file;
  runp.m_case = acase;

  tm->info() << "ConfigFile : " << config_file
             << " -- ParamFile : " << param_file << " -- Case : " << acase;

  if (!config_file.empty()) {
    if (platform::isFileReadable(config_file)) {
      JSONDocument json_doc;
      {
        UniqueArray<Byte> bytes;

        if (_readAllFile(config_file, false, bytes)) {
          ARCCORE_FATAL("Config file not available");
        }

        json_doc.parse(bytes, config_file, (JSONDocument::ParseCommentsFlag | JSONDocument::ParseNumbersAsStringsFlag));
      }

      const JSONValue root = json_doc.root();
      readArccherPart(runp, root);
    }
    else {
      ARCCORE_FATAL("Config file is not found");
    }
  }

  if (param_file.empty()) {
    ARCCORE_FATAL("Param file must be specified");
  }

  if (platform::isFileReadable(param_file)) {
    JSONDocument json_doc;
    {
      UniqueArray<Byte> bytes;

      if (_readAllFile(param_file, false, bytes)) {
        ARCCORE_FATAL("Param file not available");
      }

      json_doc.parse(bytes, param_file, (JSONDocument::ParseCommentsFlag | JSONDocument::ParseNumbersAsStringsFlag));
    }

    const JSONValue root = json_doc.root();
    readArccherPart(runp, root.child("general").child("arccher"));

    {
      const JSONValue acase_v = root.child("cases").child(acase);
      readArccherPart(runp, acase_v.child("arccher"));
    }
  }
  else {
    ARCCORE_FATAL("Param file is not found");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void createCmd(ITraceMng* tm, RunParam& runp, UniqueArray<std::string>& cmd)
{
  if (runp.m_mpi_nb_proc > 0) {
    cmd.add(runp.m_mpi_exec_path.localstr());
    cmd.add("-n");
    cmd.add( String::fromNumber(runp.m_mpi_nb_proc).localstr());
  }
  cmd.add(runp.m_app_exec_path.localstr());

  cmd.add(String::format("-A,ParamFile={0}", runp.m_param_file).localstr());
  if (!runp.m_case.empty()) {
    cmd.add(String::format("-A,Case={0}", runp.m_case).localstr());
  }

  tm->info() << "Final cmd: " << cmd;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void run(RunParam& runp, ITraceMng* tm)
{
  if (runp.m_app_exec_path.empty())
    return;

  reproc::process process;
  reproc::options options;

  // options.redirect.out.type = reproc::redirect::pipe;
  // options.redirect.err.type = reproc::redirect::pipe;

  options.env.behavior = reproc::env::type::extend;
  options.env.extra = runp.m_env_var;

  options.working_directory = runp.m_working_dir.localstr();

  UniqueArray<std::string> cmd;
  createCmd(tm, runp, cmd);

  int status = -1;
  std::error_code ec;

  // ec = process.start(cmd, options);
  // if (ec) {
  //   tm->error() << "Erreur de lancement: " << ec.message();
  //   return;
  // }
  //
  // std::tie(status, ec) = process.wait(reproc::infinite);
  // if (ec) {
  //   tm->error() << "Erreur : " << ec.message();
  //   return;
  // }

  std::tie(status, ec) = reproc::run(cmd, options);
  if (ec) {
    tm->error() << "Erreur : " << ec.message();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int main(int argc, char* argv[])
{
  Ref<ITraceMng> tm(makeRef(arccoreCreateDefaultTraceMng()));
  CommandLineArguments cla(&argc, &argv);

  RunParam runp;

  readJSON(runp, cla, tm.get());
  run(runp, tm.get());

  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
