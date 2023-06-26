using System;

namespace Arcane.Axl
{
  public static class WhereExtensions
  {
    public static string Name(this Xsd.Where where)
    {
      switch(where){
      case Xsd.Where.build:
        return "WBuild";
      case Xsd.Where.computeloop:
        return "WComputeLoop";
      case Xsd.Where.continueinit:
        return "WContinueInit";
      case Xsd.Where.exit:
        return "WExit";
      case Xsd.Where.init:
        return "WInit";
      case Xsd.Where.onmeshchanged:
        return "WOnMeshChanged";
      case Xsd.Where.onmeshrefinement:
        return "WOnMeshRefinement";
      case Xsd.Where.restore:
        return "WRestore";
      case Xsd.Where.startinit:
        return "WStartInit";
      default:
        throw new TypeUnloadedException ();
      }
    }
  }
}

