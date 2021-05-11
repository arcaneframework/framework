//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.Collections.Generic;
using System.IO;
using Scriban;
using Scriban.Runtime;

namespace Arcane.Templates
{
  class GenerateApplication
  {
      const string config_template_string =
 @"<?xml version=""1.0"" ?>
 <arcane-config code-name=""{{code_name}}"">
  <time-loops>
    <time-loop name=""{{code_name}}Loop"">
      <title>{{module_name}}</title>
      <description>Default timeloop for code {{code_name}}</description>

      <modules>
        <module name=""{{module_name}}"" need=""required"" />
      </modules>

      <entry-points where=""init"">
        <entry-point name=""{{module_name}}.StartInit"" />
      </entry-points>
      <entry-points where=""compute-loop"">
        <entry-point name=""{{module_name}}.Compute"" />
      </entry-points>
    </time-loop>
  </time-loops>
</arcane-config>
";


const string moduleaxl_template_string =
@"<?xml version=""1.0"" ?>
<module name=""{{module_name}}"" version=""1.0"">
  <description>Descripteur du module {{module_name}}</description>
  <entry-points>
    <entry-point method-name=""compute"" name=""Compute"" where=""compute-loop"" property=""none"" />
    <entry-point method-name=""startInit"" name=""StartInit"" where=""start-init"" property=""none"" />
  </entry-points>
</module>
";

const string casefile_template_string = 
@"<?xml version=""1.0""?>
<case codename=""{{code_name}}"" xml:lang=""en"" codeversion=""1.0"">
  <arcane>
    <title>Sample</title>
    <timeloop>{{code_name}}Loop</timeloop>
  </arcane>

  <meshes>
    <mesh>
      <generator name=""Cartesian2D"" >
        <nb-part-x>1</nb-part-x> 
        <nb-part-y>1</nb-part-y>
        <origin>0.0 0.0</origin>
        <x><n>20</n><length>2.0</length></x>
        <y><n>20</n><length>2.0</length></y>
      </generator>
    </mesh>
  </meshes>
</case>
";

const string modulecpp_template_string =
@"// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-

#include ""{{module_name}}_axl.h""
#include <arcane/ITimeLoopMng.h>

using namespace Arcane;

/*!
 * \brief Module {{module_name}}.
 */
class {{module_name}}Module
: public Arcane{{module_name}}Object
{
 public:
  explicit {{module_name}}Module(const ModuleBuildInfo& mbi) 
  : Arcane{{module_name}}Object(mbi) { }

 public:
  /*!
   * \brief Méthode appelée à chaque itération.
   */
  void compute() override;
  /*!
   * \brief Méthode appelée lors de l'initialisation.
   */
  void startInit() override;

  /** Retourne le numéro de version du module */
  VersionInfo versionInfo() const override { return VersionInfo(1, 0, 0); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void {{module_name}}Module::
compute()
{
  info() << ""Module {{module_name}} COMPUTE"";

  // Stop code after 10 iterations
  if (m_global_iteration()>10)
    subDomain()->timeLoopMng()->stopComputeLoop(true);
}

void {{module_name}}Module::
startInit()
{
  info() << ""Module {{module_name}} INIT"";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_{{module_name_uppercase}}({{module_name}}Module);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
";

const string maincpp_template_string =
@"// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#include <arcane/launcher/ArcaneLauncher.h>

using namespace Arcane;

int
main(int argc,char* argv[])
{
  ArcaneLauncher::init(CommandLineArguments(&argc,&argv));
  auto& app_build_info = ArcaneLauncher::applicationBuildInfo();
  app_build_info.setCodeName(""{{code_name}}"");
  app_build_info.setCodeVersion(VersionInfo(1,0,0));
  return ArcaneLauncher::run();
}
";

const string cmakelist_template_string =
@"cmake_minimum_required(VERSION 3.16)
project({{code_name}} LANGUAGES CXX)

find_package(Arcane REQUIRED)

add_executable({{code_name}} {{module_name}}Module.cc main.cc {{module_name}}_axl.h)

arcane_generate_axl({{module_name}})
arcane_add_arcane_libraries_to_target({{code_name}})
target_include_directories({{code_name}} PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
configure_file({{code_name}}.config ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
";

    static void ApplyTemplate(string string_template,TemplateContext context,string dirname,string filename)
    {
      // Parse the template
      var template = Template.Parse(string_template);

      // Check for any errors
      if (template.HasErrors) {
        foreach (var error in template.Messages) {
          Console.WriteLine(error);
        }
        throw new ApplicationException("Bad template");
      }

      var result = template.Render(context);
      //Console.WriteLine(result);
      string full_name = Path.Combine(dirname,filename);
      Console.WriteLine($"Writing file {full_name}");
      File.WriteAllText(full_name,result);
    }

    public int Execute(string[] args)
    {
      var script_object = new ScriptObject();

      string module_name = null;
      string code_name = null;
      string output_directory_name = null;

      Mono.Options.OptionSet opt_set = new Mono.Options.OptionSet();
      bool show_help = false;

      opt_set.Add("h|help|?", "display this help message", (v) => show_help = v != null);
      opt_set.Add("c|code-name=", "name of the code", (string v) => code_name = v);
      opt_set.Add("m|module-name=", "name of the module", (string v) => module_name = v);
      opt_set.Add("o|output-directory=", "directory to write files", (string v) => output_directory_name = v);
      List<string> remaining_args = opt_set.Parse(args);

      if (show_help) {
        Console.WriteLine("This script will generate a skeleton of an application for Arcane.");
        Console.WriteLine("Options:");
        opt_set.WriteOptionDescriptions(Console.Out);
        Environment.Exit(0);
      }
      if (String.IsNullOrEmpty(code_name))
        throw new ArgumentException("argument '--code-name' is missing");
      if (String.IsNullOrEmpty(module_name))
        throw new ArgumentException("argument '--module-name' is missing");
      if (String.IsNullOrEmpty(output_directory_name))
        throw new ArgumentException("argument '--output-directory' is missing");

      Console.WriteLine("Generating in directory '{0}' skeleton for code '{1}' and module '{2}' ...",
                        output_directory_name,code_name,module_name);

      script_object.Add("code_name", code_name);
      script_object.Add("module_name", module_name);
      script_object.Add("module_name_uppercase", module_name.ToUpper());

      // Applique les templates et génère les fichiers
      var context = new TemplateContext();
      context.PushGlobal(script_object);
      ApplyTemplate(config_template_string,context,output_directory_name,code_name+".config");
      ApplyTemplate(casefile_template_string,context,output_directory_name,code_name+".arc");
      ApplyTemplate(modulecpp_template_string,context,output_directory_name,module_name+"Module.cc");
      ApplyTemplate(moduleaxl_template_string,context,output_directory_name,module_name+".axl");
      ApplyTemplate(maincpp_template_string,context,output_directory_name,"main.cc");
      ApplyTemplate(cmakelist_template_string,context,output_directory_name,"CMakeLists.txt");
      return 0;
    }
  }
}
