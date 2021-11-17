using System;
using System.IO;

namespace Arcane.Axl
{
  /**
   * Classe implémentant le Design Pattern du visiteur pour générer le code C++
   * du corps du constructeur de la classe CaseOptions.
   */
  public class CppBuilderBodyGenerator : IOptionInfoVisitor
  {
    public CppBuilderBodyGenerator(TextWriter stream)
    {
      m_stream = stream;
    }
    
    public virtual void VisitComplex(ComplexOptionInfo info)
    {
    }
    
    public virtual void VisitExtended(ExtendedOptionInfo info)
    {
      string var_name = CppCodeGenerator.ToFuncName(info.Name);
      CppCodeGenerator.writeNameTranslations(m_stream,
                                             info.m_alternative_names,
                                             var_name);
    }
    
    public virtual void VisitEnumeration(EnumerationOptionInfo info)
    {
      string var_name = CppCodeGenerator.ToFuncName(info.Name);
      CppCodeGenerator.writeNameTranslations(m_stream,
                                             info.m_alternative_names,
                                             var_name);
      
      foreach(EnumValueOptionInfo ev in info.EnumValues){
        m_stream.Write("  {\n");
        m_stream.Write("    Arcane::CaseOptionEnumValue* x ");
        m_stream.Write(" = new Arcane::CaseOptionEnumValue(Arcane::String(\"");
        m_stream.Write(ev.Name + "\")");
        m_stream.Write(",(int)(" + ev.GeneratedValue + "));\n  ");
        CppCodeGenerator.writeNameTranslations(m_stream,
                                               ev.m_alternative_names,
                                               "(*x)");
        m_stream.Write("    " + CppCodeGenerator.ToFuncName(info.Name)
                       + ".addEnumValue(x,false);\n");
        m_stream.Write("  }\n");
      }
    }
    
    public virtual void VisitScript(ScriptOptionInfo info)
    {
      string var_name = CppCodeGenerator.ToFuncName(info.Name);
      CppCodeGenerator.writeNameTranslations(m_stream,
                                             info.m_alternative_names,
                                             var_name);
    }
    
    public virtual void VisitSimple(SimpleOptionInfo info)
    {
      string var_name = CppCodeGenerator.ToFuncName(info.Name);
      CppCodeGenerator.writeNameTranslations(m_stream,
                                             info.m_alternative_names,
                                             var_name);
    }
    
    public virtual void VisitServiceInstance(ServiceInstanceOptionInfo info)
    {
      string var_name = CppCodeGenerator.ToFuncName(info.Name);
      CppCodeGenerator.writeNameTranslations(m_stream,
                                             info.m_alternative_names,
                                             var_name);
    }
    
    private TextWriter m_stream;
  }
}

