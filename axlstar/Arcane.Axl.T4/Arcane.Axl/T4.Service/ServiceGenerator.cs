//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
﻿using System;
using System.Linq;
using System.IO;
using System.Collections.Generic;

namespace Arcane.Axl
{
  public class ServiceGenerator
  {
    private enum GenTarget
    {
      CaseOptionOnly,
      StrongOptionOnly,
      CaseAndStrongOption,
      Unknown 
    }

    // carefull conformity with Config Generator
    private GenTarget genTarget (string target)
    {
      if (target==null || target == "caseoption") {
        return GenTarget.CaseOptionOnly;
      } 
      if (target == "strongoption") {
        return GenTarget.StrongOptionOnly;
      }
      if (target == "all") {
        return GenTarget.CaseAndStrongOption;
      }
      Console.WriteLine ("Unknown generation option mode --gen-target arg");
      Environment.Exit (1);
      return GenTarget.Unknown;
    }


    private Arcane.Axl.Xsd.Service service;
    private string OutputPath;
    private string IncludePath;
    private string GenVersion;

    public ServiceGenerator(Arcane.Axl.Xsd.Service _service, string include_path, string output_path, string _gen_version)
    {
      service = _service;
      OutputPath = output_path;
      IncludePath = include_path;
      GenVersion = _gen_version;
    }

    // Create IOptions file common interface for CaseOptionsT and strongoptions
    private void GenerateServiceFile_IOption(bool with_arcane)
    {
      var output_IOptions = Path.Combine (OutputPath, service.Name + "_IOptions.h");
      using (var file = new StreamWriter(output_IOptions)) {
        var myIOptions = new IOptions (service, version: GenVersion, withArcane: with_arcane);
        file.WriteLine (myIOptions.TransformText ());
        var myOptionsUtils = new OptionsUtils (service);
        file.WriteLine (myOptionsUtils.TransformText ());
      }
    }

    // Create strongoptions file
    private void GenerateServiceFile_StrongOption(bool with_arcane, bool with_parameter_factory)
    {
      var output_strong_h = Path.Combine (OutputPath, service.Name + "_StrongOptions.h");
      using (var file = new StreamWriter(output_strong_h)) {
        var strong_options = new StrongOptions_h (service, version:GenVersion, withArcane: with_arcane);
        file.WriteLine (strong_options.TransformText ());
        if(with_parameter_factory){
          var strong_options_factory = new StrongOptionsFactory (service, withArcane: with_arcane);
          file.WriteLine (strong_options_factory.TransformText ());
        }
      }
    }

    // Create CaseOptionsT file
    private void GenerateServiceFile_CaseOptionsT(){
      var output_CaseOptionsT = Path.Combine (OutputPath, service.Name + "_CaseOptionsT.h");
      using (var file = new StreamWriter(output_CaseOptionsT)) {
        var myCaseOptionsT = new CaseOptionsT (service, version:GenVersion);
        file.WriteLine (myCaseOptionsT.TransformText ());
      }
    }

    private void GenerateCaseAndStrongOptionServiceFile_Axl(bool _with_mesh)
    {
      var output = Path.Combine (OutputPath, service.Name + "_axl.h");
      using (var file = new StreamWriter(output)) 
      {
        var case_options = new CaseOptionsT4 (service, IncludePath, version:GenVersion);
        file.WriteLine (case_options.TransformText ());
        var base_class = new ServiceT4CaseAndStrong (service, path:IncludePath, version:GenVersion,  withMesh:_with_mesh);
        file.WriteLine (base_class.TransformText ());
      }
    }

    private void GenerateStrongOnlyOptionServiceFile_Axl(bool _with_mesh, bool _with_arcane)
    {
      var output = Path.Combine (OutputPath, service.Name + "_axl.h");
      using (var file = new StreamWriter(output)) 
      {
        var base_class = new ServiceT4StrongOnly (service, path:IncludePath, version:GenVersion, withMesh:_with_mesh, withArcane:_with_arcane);
        file.WriteLine (base_class.TransformText ());
      }
    }

    public void GenerateStandardServiceFile_Axl()
    {
      var output = Path.Combine (OutputPath, service.Name + "_axl.h");
      using (var file = new StreamWriter(output)) 
      {
        var case_options = new CaseOptionsT4 (service, IncludePath, version:GenVersion);
        file.WriteLine (case_options.TransformText ());
        var base_class = new ServiceT4Standard (service, path:IncludePath, version:GenVersion);
        file.WriteLine (base_class.TransformText ());
      }
    }

    public void GenerateCaseOptionService(string generation_mode, bool with_mesh, bool with_arcane, bool with_parameter_factory){
      GenTarget gen_target = genTarget(generation_mode);
      switch (gen_target) {
        // Case Option only non reg cea
      case GenTarget.CaseOptionOnly :
        GenerateStandardServiceFile_Axl();
        break;
        // Strong Option only arcane light (exemple Alien)
      case GenTarget.StrongOptionOnly :
        GenerateStrongOnlyOptionServiceFile_Axl(with_mesh, with_arcane);
        GenerateServiceFile_StrongOption (with_arcane, with_parameter_factory);
        GenerateServiceFile_IOption (with_arcane);
        break;
        // Both mode
      case GenTarget.CaseAndStrongOption :
        GenerateCaseAndStrongOptionServiceFile_Axl(with_mesh);
        GenerateServiceFile_CaseOptionsT ();
        GenerateServiceFile_StrongOption (true, false);
        GenerateServiceFile_IOption (true);
        break;
      case GenTarget.Unknown :
        Console.WriteLine ("Unknown generation option mode --gen-target arg");
        Environment.Exit (1);
        break;
      }
    }
  }
}

