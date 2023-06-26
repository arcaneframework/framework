using System;
using System.Linq;
using System.Collections.Generic;

namespace Arcane.Axl.Xsd
{
  public partial class Service
  {
    public Service ()
    {
    }

    public IEnumerable<testsTest> Tests { get { return tests.test; } }

    public bool HasTests { get { return tests != null && Tests.Count () > 0; } }

    public string ParentName 
    {
      get { 
        if (String.IsNullOrEmpty(parentname)) 
        {
          if (type == ServiceTypeList.caseoption || type == ServiceTypeList.subdomain) 
            return "Arcane::BasicService";
          else
            return "Arcane::AbstractService";
        }
        return parentname; 
      }
    }

    public IEnumerable<@interface> Interfaces { get { return @interface; } } 

    public IEnumerable<@interface> InheritedInterfaces 
    {
      get { 
        return @interface.Where(p => p.inheritedSpecified ? p.inherited: true); 
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
  }
}

