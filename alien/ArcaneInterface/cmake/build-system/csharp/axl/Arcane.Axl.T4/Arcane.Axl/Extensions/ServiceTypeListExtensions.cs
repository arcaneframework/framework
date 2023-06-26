using System;

namespace Arcane.Axl
{
  public static class ServiceTypeListExtensions
  {
    public static string Name(this Xsd.ServiceTypeList type)
    {
      switch(type){
      case Xsd.ServiceTypeList.application:
        return "Application";
      case Xsd.ServiceTypeList.caseoption:
        return "CaseOption";
      case Xsd.ServiceTypeList.session:
        return "Session";
      case Xsd.ServiceTypeList.subdomain:
        return "SubDomain";
      default:
        throw new TypeUnloadedException (); 
      }
    }
  }
}

