using System;
using System.Linq;
using System.Collections;
using System.Collections.Generic;

namespace Arcane.Axl
{
  public class OptionHandler
  {
    public string Name { get; private set; }
    public string ClassName { get; private set; }
    public string NamespaceMacroName { get; private set; }
    public string NamespaceName { get; private set; }

    public IEnumerable<Xsd.Option> Options  { get; private set; }

    public IEnumerable<Xsd.Simple> Simple { get; private set; }
    public IEnumerable<Xsd.ExtendedType> Extended { get; private set; }
    public IEnumerable<Xsd.Script> Script { get; private set; }
    public IEnumerable<Xsd.ServiceInstance> ServiceInstance { get; private set; }
    public IEnumerable<Xsd.Enumeration> Enumeration { get; private set; }
    public IEnumerable<Xsd.Complex> Complex { get; private set; }
    public IEnumerable<Xsd.Attribute> Attribute { get; private set; }

    public IEnumerable<Xsd.ExtendedType> FlatteningExtended { get; private set; }
    public IEnumerable<Xsd.Complex> FlatteningComplex { get; private set; }

    public bool IsModule  { get; private set; } 
    public bool IsService { get { return !IsModule; } } 

    public bool IsNotCaseOption { get; private set; } 

    public string Version { get; private set; }
    
    public bool LesserThanVersionOne { get; private set; }

    public OptionHandler (Xsd.Module module)
    {
      NamespaceName = module.namespacename;
      NamespaceMacroName = module.namespacemacroname;
      Name = module.Name;

      IsModule = true;

      ClassName = Name;

      Version = module.versionSpecified ? "1.0" : "0.0";


      LesserThanVersionOne = !module.versionSpecified;

      if (module.options == null) {
        Options = new List<Xsd.Option> ();
      } else { 
        Options = module.options.Select(p => p as Xsd.Option);
      }

      InitializeOptions ();
    }

    public OptionHandler (Xsd.Service service)
    {
      NamespaceName = service.namespacename;
      NamespaceMacroName = service.namespacemacroname;
      Name = service.Name;

      IsModule = false;

      IsNotCaseOption = service.IsNotCaseOption;

      ClassName = Name;

      Version =  service.versionSpecified ? "1.0" : "0.0";
      
      LesserThanVersionOne = !service.versionSpecified;

      if (service.options == null) {
        Options = new List<Xsd.Option> ();
      } else { 
        Options = service.options.Select(p => p as Xsd.Option);
      }

      InitializeOptions ();
    }

    public OptionHandler (Xsd.Complex complex)
    {
      if (complex.Items == null) {
        Options = new List<Xsd.Option> ();
      } else { 
        Options = complex.Items.Select (p => p as Xsd.Option);
      }

      InitializeOptions ();
    }

    private IEnumerable<T> OptionsAs<T>()
      where T : class
    {
      return Options.Where(p => p is T).Select(p => p as T);
    }

    private void InitializeOptions ()
    {
      Simple          = OptionsAs<Xsd.Simple>();
      Extended        = OptionsAs<Xsd.ExtendedType>();
      Script          = OptionsAs<Xsd.Script>();
      ServiceInstance = OptionsAs<Xsd.ServiceInstance>();
      Enumeration     = OptionsAs<Xsd.Enumeration>();
      Complex         = OptionsAs<Xsd.Complex>();
      Attribute       = OptionsAs<Xsd.Attribute>();
     
      ComputeFlatteningExtended();
      ComputeFlatteningComplex();

      var duplicates = FlatteningComplex.Where( p => !p.IsRef)
                                        .GroupBy( x => x.type )
                                        .Where( g => g.Count() > 1 );

      if (duplicates.Any ()) {
        var duplicated_types = duplicates.Select(x => x.Key);
        String s = String.Join (",", duplicated_types);
        throw new Exception(String.Format("Duplicated complex type found for {0}", s));
      }
    }

    private void ComputeFlatteningExtended ()
    {
      var extended_list = new List<Xsd.ExtendedType>();

      extended_list.AddRange(Extended);

      foreach(var complex in Complex) {
        extended_list.AddRange(complex.Extended);
      }

      FlatteningExtended = extended_list;
    }

    private void ComputeFlatteningComplex ()
    {
      var complex_list = new List<Xsd.Complex>();

      foreach (var complex in Complex)
      {
        complex_list.AddRange(complex.FlatteningComplex);
        complex_list.Add (complex);
      }

      FlatteningComplex = complex_list.Distinct ();
    }
  }
}

