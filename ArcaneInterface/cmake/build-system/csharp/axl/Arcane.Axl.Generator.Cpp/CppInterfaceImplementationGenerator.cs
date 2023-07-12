using System;
using System.IO;

namespace Arcane.Axl
{
  /**
   * Visiteur pour implémenter les méthodes lorsqu'une option dérive d'une interface.
   */
  public class CppInterfaceImplementationGenerator : IOptionInfoVisitor
  {
    public CppInterfaceImplementationGenerator (TextWriter stream)
    {
      m_stream = stream;
    }

    public virtual void VisitComplex(ComplexOptionInfo info)
    {
      //Console.WriteLine("VISIT IMPLEMENTATION opt='{0}'",info.Name);
      ComplexOptionInfo ref_info = info;
      if (info.ReferenceType!=null)
        ref_info = info.ReferenceType;

      string it_type = ref_info.InterfaceType;
      // Si l'option fait référence à une autre option, utilise le
      // nom de l'interface de l'autre option
      //if (info.ReferenceType!=null)
        //it_type = info.ReferenceType.InterfaceType;
      string interface_type = CppUtils.ConvertType(it_type);
      if (String.IsNullOrEmpty(interface_type))
        // Ne génère rien dans ce cas 
        return;

      if (ref_info.IsMulti){
        string instance_name = CppCodeGenerator.ToFuncName(info.Name);
        string func_name = CppCodeGenerator.ToClassName(info.Name);
        m_stream.Write ("  virtual Arcane::ConstArrayView< "+interface_type+"*> get"+func_name+"(){ return "+instance_name+"._interface(); }\n");
      }
      else{
        string return_type_name = interface_type + "*";
        string instance_name = CppCodeGenerator.ToFuncName(info.Name);
        string func_name = CppCodeGenerator.ToClassName(info.Name);
        m_stream.Write ("  virtual "+return_type_name+" get"+func_name+"(){ return "+instance_name+"._interface(); }\n");
        if (info.IsOptional) {
          m_stream.Write ("  virtual bool has"+func_name+"() const { return "+instance_name+".isPresent(); }\n");
        }
      }
    }
    public virtual void VisitExtended(ExtendedOptionInfo info)
    {
      string qtn = CppUtils.ConvertType(info.Type);
      if (info.IsMulti){
        _WriteMethod("Arcane::ConstArrayView< "+qtn+ " >",info.Name,"",false);
      }
      else
        _WriteMethod(qtn,info.Name,"()",info.IsOptional);
    }
    public virtual void VisitEnumeration(EnumerationOptionInfo info)
    {
      string qtn = CppUtils.ConvertType(info.Type);
      if (info.IsMulti){
        _WriteMethod("Arcane::ConstArrayView< "+qtn+ " >",info.Name,"",false);
      }
      else
        _WriteMethod(qtn,info.Name,"()",info.IsOptional);
    }
    public virtual void VisitScript(ScriptOptionInfo info) { }
    public virtual void VisitSimple(SimpleOptionInfo info)
    {
      if (info.IsMulti) {
        string qtn = CppUtils.BasicTypeQualifiedName(info.SimpleType);
        _WriteMethod ("Arcane::ConstArrayView< " + qtn + " >", info.Name, "()", false);
      } else {
        string qtn = CppUtils.ReturnTypeQualifiedName (info.SimpleType);
        _WriteMethod (qtn, info.Name, "()", info.IsOptional);
      }
    }

    public virtual void VisitServiceInstance(ServiceInstanceOptionInfo info)
    {
      string qtn = CppUtils.ConvertType(info.Type)+"*";
      if (info.IsMulti){
        _WriteMethod("Arcane::ConstArrayView< "+qtn+ " >",info.Name,"",false);
      }
      else
        _WriteMethod(qtn,info.Name,"()",info.IsOptional);
    }

    void _WriteMethod(string return_type_name,string option_name,string call_op,bool is_optional)
    {
      string instance_name = CppCodeGenerator.ToFuncName(option_name);
      string func_name = CppCodeGenerator.ToClassName(option_name);
      m_stream.Write ("  virtual "+return_type_name+" get"+func_name+"(){ return "+instance_name+call_op+"; }\n");
      if (is_optional)
        m_stream.Write ("  virtual bool has"+func_name+"() const { return "+instance_name+".isPresent(); }\n");
    }

    private TextWriter m_stream;
  }
}

