using System;
using System.Collections.Generic;

namespace Arcane.Axl.Xsd
{
  public partial class variablesVariable
  {
    public enum Property {
      NoDump, 
      NoNeedSync, 
      ExecutionDepend, 
      SubDomainDepend,
      SubDomainPrivate,
      NoRestore,
      Undefined
    }

    public variablesVariable ()
    {
    }

    public string ClassName {
      get {
        string name = "Variable";
        if (itemkind != ItemKind.none)
          name += itemkind.Name ();
        switch (int.Parse(dim)) {
        case 0:
          if (itemkind == ItemKind.none)
            name += "Scalar";
          break;
        case 1:
          name += "Array";
          break;
        case 2:
          name += "Array2";
          break;
        }
        name += datatype.Name ();
        return name;
      }
    }

    public bool IsNoDump { get { return dumpSpecified ? !dump : true; } }

    public bool IsNoNeedSync { get { return needsyncSpecified ? !needsync : true; } }

    public bool IsExecutionDepend { get { return executiondependSpecified ? executiondepend : false; } }

    public bool IsSubDomainDepend { get { return subdomaindependSpecified ? subdomaindepend : false; } }

    public bool IsSubDomainPrivate { get { return subdomainprivateSpecified ? subdomainprivate : false; } }

    public bool IsInFlow { get { return flow == Flow.@in; } }

    public bool IsNoRestore { 
      get { 
        if (GlobalContext.Instance.NoRestore == true) {
          return true;
        } else {
          return norestoreSpecified ? norestore : false; 
        }
      } 
    }

    public int NbProperty { 
      get { 
        return Convert.ToInt32(IsNoDump) 
          + Convert.ToInt32(IsNoNeedSync) 
          + Convert.ToInt32(IsExecutionDepend) 
          + Convert.ToInt32(IsSubDomainDepend)
          + Convert.ToInt32(IsSubDomainPrivate)
          + Convert.ToInt32(IsNoRestore); 
      }
    }

    public Property FirstProperty { 
      get { 
        if (IsNoDump)
          return Property.NoDump;
        if (IsNoNeedSync)
          return Property.NoNeedSync;
        if (IsExecutionDepend)
          return Property.ExecutionDepend;
        if (IsSubDomainDepend)
          return Property.SubDomainDepend;
        if (IsSubDomainPrivate)
          return Property.SubDomainPrivate;
        if (IsNoRestore)
          return Property.NoRestore;
        return Property.Undefined;
      } 
    }

    public IEnumerable<Property> OthersProperties { 
      get { 
        var properties = new List<Property> ();
        if (IsNoDump)
          properties.Add (Property.NoDump);
        if (IsNoNeedSync)
          properties.Add (Property.NoNeedSync);
        if (IsExecutionDepend)
          properties.Add (Property.ExecutionDepend);
        if (IsSubDomainDepend)
          properties.Add (Property.SubDomainDepend);
        if (IsSubDomainPrivate)
          properties.Add (Property.SubDomainPrivate);
        if (IsNoRestore)
          properties.Add (Property.NoRestore);
        properties.RemoveAt (0);
        return properties;
      } 
    }
  }
}

