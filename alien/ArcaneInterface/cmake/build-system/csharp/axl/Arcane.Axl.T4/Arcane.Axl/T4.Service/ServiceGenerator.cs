using System;
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
    private string output_path;
    private string include_path;
    private string gen_version;

    public ServiceGenerator(Arcane.Axl.Xsd.Service _service, string _output_path, string _include_path, string _gen_version)
    {
      service = _service;
      output_path = _output_path;
      include_path = _include_path;
      gen_version = _gen_version;
    }

    // Create IOptions file common interface for CaseOptionsT and strongoptions
    private void GenerateServiceFile_IOption(bool with_arcane)
    {
      var output_IOptions = Path.Combine (output_path, service.Name + "_IOptions.h");
      using (var file = new StreamWriter(output_IOptions)) {
        var myIOptions = new IOptions (service, version: gen_version, withArcane: with_arcane);
        file.WriteLine (myIOptions.TransformText ());
        var myOptionsUtils = new OptionsUtils (service);
        file.WriteLine (myOptionsUtils.TransformText ());
      }
    }

    // Create strongoptions file
    private void GenerateServiceFile_StrongOption(bool with_arcane, bool with_parameter_factory)
    {
      var output_strong_h = Path.Combine (output_path, service.Name + "_StrongOptions.h");
      using (var file = new StreamWriter(output_strong_h)) {
        var strong_options = new StrongOptions_h (service, version:gen_version, withArcane: with_arcane);
        file.WriteLine (strong_options.TransformText ());
        if(with_parameter_factory){
          var strong_options_factory = new StrongOptionsFactory (service, withArcane: with_arcane);
          file.WriteLine (strong_options_factory.TransformText ());
        }
      }
    }

    // Create CaseOptionsT file
    private void GenerateServiceFile_CaseOptionsT(){
      var output_CaseOptionsT = Path.Combine (output_path, service.Name + "_CaseOptionsT.h");
      using (var file = new StreamWriter(output_CaseOptionsT)) {
        var myCaseOptionsT = new CaseOptionsT (service, version:gen_version);
        file.WriteLine (myCaseOptionsT.TransformText ());
      }
    }

    private void GenerateCaseAndStrongOptionServiceFile_Axl(bool _with_mesh)
    {
      var output = Path.Combine (output_path, service.Name + "_axl.h");
      using (var file = new StreamWriter(output)) 
      {
        var case_options = new CaseOptionsT4 (service, version:gen_version);
        file.WriteLine (case_options.TransformText ());
        var base_class = new ServiceT4CaseAndStrong (service, path:include_path, version:gen_version,  withMesh:_with_mesh);
        file.WriteLine (base_class.TransformText ());
      }
    }

    private void GenerateStrongOnlyOptionServiceFile_Axl(bool _with_mesh, bool _with_arcane)
    {
      var output = Path.Combine (output_path, service.Name + "_axl.h");
      using (var file = new StreamWriter(output)) 
      {
        var base_class = new ServiceT4StrongOnly (service, path:include_path, version:gen_version, withMesh:_with_mesh, withArcane:_with_arcane);
        file.WriteLine (base_class.TransformText ());
      }
    }

    public void GenerateStandardServiceFile_Axl()
    {
      var output = Path.Combine (output_path, service.Name + "_axl.h");
      using (var file = new StreamWriter(output)) 
      {
        var case_options = new CaseOptionsT4 (service, version:gen_version);
        file.WriteLine (case_options.TransformText ());
        var base_class = new ServiceT4Standard (service, path:include_path, version:gen_version);
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

