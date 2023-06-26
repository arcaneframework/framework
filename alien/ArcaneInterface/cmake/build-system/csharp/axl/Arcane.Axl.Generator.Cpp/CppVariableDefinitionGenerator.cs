using System;
using System.IO;

namespace Arcane.Axl
{
  /**
   * Classe implémentant le Design Pattern du visiteur pour générer le code C++ de
   * déclaration des variables dans le fichier .h.
   */
  public class CppVariableDefinitionGenerator : IOptionInfoVisitor
  {
    public CppVariableDefinitionGenerator(TextWriter stream)
    {
      m_stream = stream;
    }
    
    public virtual void VisitComplex(ComplexOptionInfo info)
    {
      if (info.Name != null)
        if (info.Type == null)
          _visitOptionInfo(info, info.ReferenceTypeName, false);
      else
        _visitOptionInfo(info, info.Type, false);
    }
    
    public virtual void VisitExtended(ExtendedOptionInfo info)
    {
      string s;
      if (info.IsMulti)
        s = "MultiExtendedT< " + info.Type + " >";
      else
        s = "ExtendedT< " + info.Type + " >";
      _visitOptionInfo(info, s);
    }
    
    public virtual void VisitEnumeration(EnumerationOptionInfo info)
    {
      string s;
      if (info.IsMulti)
        s = "MultiEnumT< " + info.Type + " >";
      else
        s = "EnumT< " + info.Type + " >";
      _visitOptionInfo(info, s);
    }
    
    public virtual void VisitScript(ScriptOptionInfo info)
    {
      _visitOptionInfo(info, "Script");
    }
    
    public virtual void VisitSimple(SimpleOptionInfo info)
    {
      string typeT_name = string.Empty;
      string class_name = CppUtils.BasicTypeName(info.SimpleType);
      if (info.SimpleType==SimpleOptionInfoType.TypeBool){
        typeT_name = "bool";
      }

      if (info.IsMulti)
        if (String.IsNullOrEmpty(typeT_name))
        m_stream.Write("Arcane::CaseOptionMultiSimpleT< Arcane::{0} >   {1};\n",
                         class_name, CppCodeGenerator.ToFuncName(info.Name));
      else
        m_stream.Write("Arcane::CaseOptionMultiSimpleT< {0} >   {1};\n",
                       typeT_name, CppCodeGenerator.ToFuncName(info.Name));
      else
        _visitOptionInfo(info, class_name);
    }
    
    public virtual void VisitServiceInstance(ServiceInstanceOptionInfo info)
    {
      string cname;
      if (info.IsMulti) {
        cname = "MultiServiceT< ";
      }
      else {
        cname = "ServiceT< ";
      }
      cname += info.Type;
      cname += " >";
      _visitOptionInfo(info, cname);
    }
    
    private void _visitOptionInfo(Option info, string class_name)
    {
      _visitOptionInfo(info, class_name, true);
    }
    private void _visitOptionInfo(Option info, string class_name, bool is_built_in)
    {
      if (is_built_in)
        m_stream.Write("  Arcane::CaseOption");
      else
        m_stream.Write("  CaseOption");
      m_stream.Write(class_name + "   " + CppCodeGenerator.ToFuncName(info.Name) + ";\n");
    }
    
    private TextWriter m_stream;
  }
}

