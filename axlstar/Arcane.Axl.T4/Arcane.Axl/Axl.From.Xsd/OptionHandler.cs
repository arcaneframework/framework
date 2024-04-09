//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
﻿using System;
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

    /*!
     * Indique pour les anciens cas (version<1) s'il faut inclure un fichier Types*.h
     * et si le module ou le service dérivent d'une classe 'Types*.h'.
     */
    public bool NeedTypeInclude { get; private set; }

    public string CppHeaderGuardName
    {
      get
      {
        string base_name = "AXLSTAR";
        if (!String.IsNullOrEmpty(NamespaceName))
          base_name += "_" + NamespaceName.ToUpper().Replace(":","_");
        else if (!String.IsNullOrEmpty(NamespaceMacroName))
          base_name += "_" + NamespaceMacroName;
        return base_name + "_" + ClassName.ToUpper();
      }
    }
    public OptionHandler (Xsd.Module module)
    {
      NamespaceName = module.namespacename;
      NamespaceMacroName = module.namespacemacroname;
      Name = module.Name;

      IsModule = true;

      ClassName = Name;

      Version = module.versionSpecified ? "1.0" : "0.0";

      LesserThanVersionOne = !module.versionSpecified;

      if (module.options == null || module.options.Items == null) {
        Options = new List<Xsd.Option> ();
      } else {
        Options = module.options.Items;
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
      if (service.options==null || service.options.Items == null) {
        Options = new List<Xsd.Option> ();
      } else {
        Options = service.options.Items;
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

      if (LesserThanVersionOne) {
        bool has_namespace = !string.IsNullOrEmpty (NamespaceMacroName);
        NeedTypeInclude = (has_namespace || _GetNbExtendedAndNbEnumeration()!=0);
        //Console.WriteLine ("NEED TYPE INCLUDE=? {0}", NeedTypeInclude);
      }
    }

    int _GetNbExtendedAndNbEnumeration ()
    {
      int total = 0;
      foreach (var x in FlatteningExtended)
        ++total;
      foreach (var c in FlatteningComplex)
        total += c.Xml.Enumeration.Count ();
      total += this.Enumeration.Count ();
      return total;
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

