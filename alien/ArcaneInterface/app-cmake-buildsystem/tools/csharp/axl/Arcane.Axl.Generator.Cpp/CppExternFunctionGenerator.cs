using System;
using System.IO;

namespace Arcane.Axl
{
  /**
   * Classe implémentant le Design Pattern du visiteur pour générer le code C++ de
   * la liste des fonctions externes au début du fichier .h.
   */
  public class CppExternFunctionGenerator : IOptionInfoVisitor
  {
    public CppExternFunctionGenerator(TextWriter stream)
    {
      m_stream = stream;
    }
    
    public virtual void VisitComplex(ComplexOptionInfo info)
    {
      info.AcceptChildren(this);
    }
    
    public virtual void VisitExtended(ExtendedOptionInfo info)
    {
      m_stream.Write("extern \"C++\" bool\n");
      m_stream.Write("_caseOptionConvert(");
      m_stream.Write("const Arcane::CaseOptionBase&,const Arcane::String&,");
      m_stream.Write(info.Type + "&);\n");
    }
    
    public virtual void VisitEnumeration(EnumerationOptionInfo info) { }
    public virtual void VisitScript(ScriptOptionInfo info) { }
    public virtual void VisitSimple(SimpleOptionInfo info) { }
    public virtual void VisitServiceInstance(ServiceInstanceOptionInfo info) { }
    
    private TextWriter m_stream;
  }
}

