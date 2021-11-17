namespace Arcane.Axl
{
  /**
   * Interface du Design Pattern du visiteur pour les classes
   * d'options du fichier AXL. 
   */
  public interface IOptionInfoVisitor
  {
    void VisitComplex(ComplexOptionInfo o);
    void VisitEnumeration(EnumerationOptionInfo o);
    void VisitExtended(ExtendedOptionInfo o);
    void VisitScript(ScriptOptionInfo o);
    void VisitSimple(SimpleOptionInfo o);
    void VisitServiceInstance(ServiceInstanceOptionInfo o);
  }
}

