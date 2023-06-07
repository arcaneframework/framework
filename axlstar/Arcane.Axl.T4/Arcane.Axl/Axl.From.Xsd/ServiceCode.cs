//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.Linq;
using System.Collections.Generic;
using Newtonsoft.Json;
using System.Text;

namespace Arcane.Axl.Xsd
{
  public partial class Service
  {
    [JsonIgnore ()]
    public IEnumerable<testsTest> Tests { get { return tests.test; } }

    public bool HasTests { get { return tests != null && Tests.Count () > 0; } }

    public string ParentName
    {
      get {
        if (String.IsNullOrEmpty(parentname))
        {
          if (_HasType(ServiceTypeFlags.Mesh))
            return "Arcane::BasicMeshService";
          else if (_HasType(ServiceTypeFlags.CaseOption))
            return "Arcane::BasicCaseOptionService";
          else if (_HasType(ServiceTypeFlags.SubDomain))
            return "Arcane::BasicSubDomainService";
          else
            return "Arcane::AbstractService";
        }
        return parentname;
      }
    }

    public IEnumerable<@interface> Interfaces { get { return @interface ?? new @interface[0]; } }

    public IEnumerable<@interface> InheritedInterfaces
    {
      get {
        return Interfaces.Where(p => p.inheritedSpecified ? p.inherited: true);
      }
    }

    public bool IsNotCaseOption {
      get {
        return type != ServiceTypeList.caseoption;
      }
    }

    public IEnumerable<Xsd.variablesVariable> Variables {
      get {
        if (variables == null)
          return new List<Xsd.variablesVariable> ();
        else
          return new List<Xsd.variablesVariable> (variables);
      }
    }

    public bool HasMultiDimVariable {
      get {
        foreach (var v in Variables)
          if (v.HasShapeDim)
            return true;
        return false;
      }
    }

    public bool IsSingleton {
      get {
        return singleton;
      }
    }
    public void CheckValid ()
    {
      _CreateTypesList();
      foreach (var v in Variables)
        v.InitializeAndValidate ();
    }

    ServiceTypeFlags m_service_type_flags = 0;

    //! Flags pour savoir quels types supporte un service
    [Flags]
    enum ServiceTypeFlags
    {
       Application = 1,
       Session = 2,
       SubDomain = 4,
       CaseOption = 8,
       Mesh = 16
    };
    void _AddType(ServiceTypeList t)
    {
      Console.WriteLine("Add type {0}",t);
      switch(t){
        case ServiceTypeList.mesh:
          m_service_type_flags |= ServiceTypeFlags.Mesh;
          break;
        case ServiceTypeList.caseoption:
          m_service_type_flags |= ServiceTypeFlags.CaseOption;
          break;
        case ServiceTypeList.subdomain:
          m_service_type_flags |= ServiceTypeFlags.SubDomain;
          break;
        case ServiceTypeList.session:
          m_service_type_flags |= ServiceTypeFlags.Session;
          break;
        case ServiceTypeList.application:
          m_service_type_flags |= ServiceTypeFlags.Application;
          break;
        default:
          throw new ApplicationException($"Unknown type {t}");
      }
    }
    void _CreateTypesList()
    {
      m_service_type_flags = 0;
      _AddType(this.type);
      if (this.type2Specified)
        _AddType(this.type2);
      if (this.type3Specified)
        _AddType(this.type3);
      if (this.type4Specified)
        _AddType(this.type4);
      if (this.type5Specified)
        _AddType(this.type5);
      if (m_service_type_flags==0)
        _ThrowInvalid("No valid 'type' attribute");
    }
    void _ThrowInvalid (string msg)
    {
      var full_msg = String.Format ("{0} (service_name={1}) ", msg, this.name);
      throw new ApplicationException (full_msg);
    }
    /*!
     * \brief Converti les types en une chaîne de caractères pour
     * la macro ARCANE_REGISTER_SERVICE.
     *
     * Par exemple si on a le type ServiceTypeFlag.CaseOption et
     * et ServiceTypeFlags.SubDomain, cela retournera la chaîne suivante:
     * Arcane::ST_SubDomain|Arcane::ST_CaseOption.
    */
    public string TypesToArcaneNames()
    {
      List<String> names = new List<string>();
      if (_HasType(ServiceTypeFlags.Application))
        names.Add("Arcane::ST_Application");
      if (_HasType(ServiceTypeFlags.Session))
        names.Add("Arcane::ST_Session");
      if (_HasType(ServiceTypeFlags.SubDomain))
        names.Add("Arcane::ST_SubDomain");
      if (_HasType(ServiceTypeFlags.CaseOption))
        names.Add("Arcane::ST_CaseOption");
      if (_HasType(ServiceTypeFlags.Mesh))
        names.Add("Arcane::ST_Mesh");
      return String.Join("|",names);
    }
    bool _HasType(ServiceTypeFlags f)
    {
      return (m_service_type_flags & f) != 0;
    }
  }
}

