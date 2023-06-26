/*---------------------------------------------------------------------------*/
/* GeneratorVisitor.h                                          (C) 2000-2007 */
/*                                                                           */
/* Classes implémentant le Design Pattern du visiteur pour générer des       */
/* morceaux de code C#.                                                      */
/*---------------------------------------------------------------------------*/
using System;
using System.IO;
using System.Collections.Generic;

namespace Arcane.Axl
{
  /**
   * Classe implémentant le Design Pattern du visiteur pour générer le code C#
   * du corps du contructeur de la classe CaseOptions.
   */
  public class CSharpBuilderBodyGenerator : IOptionInfoVisitor
  {
    public CSharpBuilderBodyGenerator(TextWriter stream)
    {
      m_stream = stream;
    }

    public virtual void VisitComplex(ComplexOptionInfo info)
    {
    }

    public virtual void VisitExtended(ExtendedOptionInfo info)
    {
      string var_name = CSharpCodeGenerator.ToFuncName(info.Name);
      CSharpCodeGenerator.writeNameTranslations(m_stream,
                                           info.m_alternative_names,
                                           var_name);
    }

    public virtual void VisitEnumeration(EnumerationOptionInfo info)
    {
      string var_name = CSharpCodeGenerator.ToFuncName(info.Name);
      CSharpCodeGenerator.writeNameTranslations(m_stream,
                                           info.m_alternative_names,
                                           var_name);

      foreach(EnumValueOptionInfo ev in info.EnumValues){
        m_stream.Write("  {\n");
        m_stream.Write("    Arcane.CaseOptionEnumValue x ");
        m_stream.Write(" = new Arcane.CaseOptionEnumValue(\"");
        m_stream.Write(ev.Name + "\"");
        // Supprime le nom du type du axl dans la valeur
        // Par exemple Test::Value devient 'Value'
        string gen_value = ev.GeneratedValue;
        gen_value = gen_value.Replace("::",".");
        int index = gen_value.LastIndexOf(".");
        if (index>0)
          gen_value = gen_value.Substring(index+1);
        string type_value = CSharpCodeGenerator.ToCSharpType(info.Type+"."+gen_value);
        m_stream.Write(",(int)(" + type_value + "));\n  ");
        CSharpCodeGenerator.writeNameTranslations(m_stream,
                                            ev.m_alternative_names,
                                            "(*x)");
        m_stream.Write("    m_" + CSharpCodeGenerator.ToFuncName(info.Name)
                       + ".addEnumValue(x,true);\n");
        m_stream.Write("  }\n");
      }
    }

    public virtual void VisitScript(ScriptOptionInfo info)
    {
      string var_name = CSharpCodeGenerator.ToFuncName(info.Name);
      CSharpCodeGenerator.writeNameTranslations(m_stream,
                                           info.m_alternative_names,
                                           var_name);
    }

    public virtual void VisitSimple(SimpleOptionInfo info)
    {
      string var_name = CSharpCodeGenerator.ToFuncName(info.Name);
      CSharpCodeGenerator.writeNameTranslations(m_stream,
                                           info.m_alternative_names,
                                           var_name);
    }

    public virtual void VisitServiceInstance(ServiceInstanceOptionInfo info)
    {
      string var_name = CSharpCodeGenerator.ToFuncName(info.Name);
      CSharpCodeGenerator.writeNameTranslations(m_stream,
                                           info.m_alternative_names,
                                           var_name);
    }

    private TextWriter m_stream;
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  /**
   * Classe implémentant le Design Pattern du visiteur pour générer le code C++
   * se trouvant avant le corps du ructeur dans le fichier .cc.
   */
  public class CSharpOptionBuilderGenerator : IOptionInfoVisitor
  {
    public CSharpOptionBuilderGenerator(TextWriter stream,
                                  string parent_list,
                                  string pos)
    {
      m_stream = stream;
      m_parent_list = parent_list;
      m_pos = pos;
    }

    public virtual void VisitComplex(ComplexOptionInfo info)
    {
      if (info.Name != null) {
        string var_name = CSharpCodeGenerator.ToFuncName(info.Name);
        m_stream.Write("  m_" + var_name + " = new ");
        m_stream.Write(CSharpVariableTypeGenerator.TypeName(info));
        m_stream.Write("(cm," + m_parent_list);
        m_stream.Write(",\"" + info.Name + "\"");
        m_stream.Write(", " + m_pos);
        m_stream.Write(");\n");
      }
    }

    public virtual void VisitExtended(ExtendedOptionInfo info)
    {
      _visitOptionInfo(info);
      m_stream.Write(",\"" + info.Type + "\"");
      //m_stream.Write(", new "+CSharpCodeGenerator.ToCSharpType(info.Type)+"ExtendedConverter(cm)");
      m_stream.Write(");\n");
    }

    public virtual void VisitEnumeration(EnumerationOptionInfo info)
    {
      _visitOptionInfo(info);
      m_stream.Write(",\"" + info.Type + "\"");
      m_stream.Write(");\n");
    }

    public virtual void VisitScript(ScriptOptionInfo info)
    {
      _visitOptionInfo(info);
      m_stream.Write(");\n");
    }

    public virtual void VisitSimple(SimpleOptionInfo info)
    {
      _visitOptionInfo(info);
      string unit = info.PhysicalUnit;
      m_stream.Write(',');
      if (unit!=null)
        m_stream.Write("null");
      else
        m_stream.Write('"'+unit+'"');
      m_stream.Write(");\n");
    }

    public virtual void VisitServiceInstance(ServiceInstanceOptionInfo info)
    {
      _visitOptionInfo(info);
      m_stream.Write("," + (info.AllowNull ? "true" : "false"));
      if (!info.IsMulti)
        m_stream.Write("," + (info.IsOptional ? "true" : "false"));
      m_stream.Write(");\n");
    }

    private void _visitOptionInfo(Option info)
    {
      string var_name = CSharpCodeGenerator.ToFuncName(info.Name);
      m_stream.Write(" m_" + var_name + " = new ");
      m_stream.Write(CSharpVariableTypeGenerator.TypeName(info));
      m_stream.Write("(new Arcane.CaseOptionBuildInfo(cm,");
      m_stream.Write(m_parent_list);
      m_stream.Write(",\"" + info.Name + "\"," + m_pos);
      m_stream.Write(",");
      if (info.HasDefault)
        m_stream.Write('"' + info.DefaultValue + '"');
      else
        m_stream.Write("null");
      m_stream.Write("," + info.MinOccurs);
      m_stream.Write("," + info.MaxOccurs);
      if (info.IsOptional)
        m_stream.Write(",true");
      m_stream.Write(")");
    }

    private TextWriter m_stream;
    private string m_parent_list;
    private string m_pos;
    //private CSharpVariableTypeGenerator m_type_visitor;
  };

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * Classe implémentant le Design Pattern du visiteur pour générer le nom
   * du type de la variable.
   */
  public class CSharpVariableTypeGenerator : IOptionInfoVisitor
  {
    class SimpleTypeName
    {
      internal SimpleTypeName(string element_name,string container_name)
      {
        m_element_name = element_name;
        m_container_name = container_name;
      }
      internal string m_element_name;
      internal string m_container_name;
    }
    static Dictionary<SimpleOptionInfoType,SimpleTypeName> m_simple_option_types_name;

    static void _AddType(SimpleOptionInfoType type,string element_name,string container_name)
    {
      m_simple_option_types_name.Add(type,new SimpleTypeName(element_name,container_name));
    }

    static void _buildTypesName()
    {
      m_simple_option_types_name = new Dictionary<SimpleOptionInfoType,SimpleTypeName>();
      _AddType(SimpleOptionInfoType.TypeString,"string","String");
      _AddType(SimpleOptionInfoType.TypeBool,"bool","Bool");
      _AddType(SimpleOptionInfoType.TypeReal,"Real","Real");
      _AddType(SimpleOptionInfoType.TypeReal2,"Arcane.Real2","Real2");
      _AddType(SimpleOptionInfoType.TypeReal3,"Arcane.Real3","Real3");
      _AddType(SimpleOptionInfoType.TypeReal2x2,"Arcane.Real2x2","Real2x2");
      _AddType(SimpleOptionInfoType.TypeReal3x3,"Arcane.Real3x3","Real3x3");
      _AddType(SimpleOptionInfoType.TypeInteger,"Integer","Integer");
      _AddType(SimpleOptionInfoType.TypeInt32,"int","Int32");
      _AddType(SimpleOptionInfoType.TypeInt64,"long","Int64");
      _AddType(SimpleOptionInfoType.TypeStringArray,"string[]","Arcane.StringArray");
      _AddType(SimpleOptionInfoType.TypeBoolArray,"bool[]","Arcane.BoolArray");
      _AddType(SimpleOptionInfoType.TypeRealArray,"Real[]","Arcane.RealArray");
      _AddType(SimpleOptionInfoType.TypeReal2Array,"Arcane.Real2[]","Arcane.Real2Array");
      _AddType(SimpleOptionInfoType.TypeReal3Array,"Arcane.Real3[]","Arcane.Real3Array");
      _AddType(SimpleOptionInfoType.TypeReal2x2Array,"Arcane.Real2x2[]","Arcane.Real2x2Array");
      _AddType(SimpleOptionInfoType.TypeReal3x3Array,"Arcane.Real3x3[]","Arcane.Real3x3Array");
      _AddType(SimpleOptionInfoType.TypeIntegerArray,"Integer[]","Arcane.IntegerArray");
      _AddType(SimpleOptionInfoType.TypeInt32Array,"int[]","Arcane.Int32Array");
      _AddType(SimpleOptionInfoType.TypeInt64Array,"long[]","Arcane.Int64Array");
    }

    public static string ElementTypeName(SimpleOptionInfoType type)
    {
      if (m_simple_option_types_name==null)
        _buildTypesName();
      return m_simple_option_types_name[type].m_element_name;
    }

    public static string ContainerTypeName(SimpleOptionInfoType type)
    {
      if (m_simple_option_types_name==null)
        _buildTypesName();
      return m_simple_option_types_name[type].m_container_name;
    }
    
    private CSharpVariableTypeGenerator()
    {
    }

    static public string TypeName(Option opt)
    {
      CSharpVariableTypeGenerator type_gen = new CSharpVariableTypeGenerator();
      opt.Accept(type_gen);
      return type_gen.m_type_name.Replace("::",".");
    }

    private string m_type_name;
    //public string TypeName { get { return m_type_name; } }

    void IOptionInfoVisitor.VisitComplex(ComplexOptionInfo info)
    {
      Console.WriteLine("VISIT COMPLEX opt={0} type={1} name={2}",info,info.Type,info.Name);
      if (info.Name != null)
        if (info.Type == null)
          _visitOptionInfo(info, info.ReferenceTypeName);
        else
          _visitOptionInfo(info, "CaseOption"+info.Type);
      Console.WriteLine("VISIT COMPLEX NAME={0}",m_type_name);
    }

    void IOptionInfoVisitor.VisitExtended(ExtendedOptionInfo info)
    {
      string full_type = CSharpCodeGenerator.ToCSharpType(info.Type);
      string type_name;
      if (info.IsMulti)
        type_name = full_type + "MultiExtendedCaseOption";
      else
        type_name = full_type + "ExtendedCaseOption";
      _visitOptionInfo(info,type_name);
    }

    void IOptionInfoVisitor.VisitEnumeration(EnumerationOptionInfo info)
    {
      string s;
      string type = CSharpCodeGenerator.ToCSharpType(info.Type);
      if (info.IsMulti)
        s = "Arcane.CaseOptionMultiEnumT<" + type + ">";
      else
        s = "Arcane.CaseOptionEnumT<" + type + ">";
      _visitOptionInfo(info, s);
    }

    void IOptionInfoVisitor.VisitScript(ScriptOptionInfo info)
    {
      _visitOptionInfo(info, "Arcane.CaseOptionScript");
    }

    void IOptionInfoVisitor.VisitSimple(SimpleOptionInfo info)
    {
      string class_name = ContainerTypeName(info.SimpleType);
      if (info.IsMulti)
        m_type_name = "Arcane.CaseOptionMultiSimple"+class_name;
      else
        _visitOptionInfo(info, "Arcane.CaseOption"+class_name);
    }

    void IOptionInfoVisitor.VisitServiceInstance(ServiceInstanceOptionInfo info)
    {
      string cname;
      if (info.IsMulti) {
        cname = "Arcane.CaseOptionMultiServiceT<";
      }
      else {
        cname = "Arcane.CaseOptionServiceT<";
      }
      cname += info.Type;
      cname += ">";
      _visitOptionInfo(info, cname);
    }

    void _visitOptionInfo(Option info, string class_name)
    {
      m_type_name = class_name;
    }
  };

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * Classe implémentant le Design Pattern du visiteur pour générer la
   * définition et les accesseurs de la variable.
   */
  public class CSharpVariableDefinitionGenerator : IOptionInfoVisitor
  {
    //CSharpVariableTypeGenerator m_type_generator = new CSharpVariableTypeGenerator();
    TextWriter m_stream;
    public CSharpVariableDefinitionGenerator(TextWriter stream)
    {
      m_stream = stream;
    }

    public virtual void VisitComplex(ComplexOptionInfo opt)
    {
      string field_type_name = _FieldTypeName(opt);
      Console.WriteLine("COMPLEX FIELD TYPE NAME ={0}",field_type_name);
      _WriteFieldName(opt,field_type_name);
      string field_name = CSharpCodeGenerator.ToFuncName(opt.Name);
      string property_name = CSharpCodeGenerator.ToClassName(field_name);
      m_stream.Write("  public ");
      m_stream.Write(field_type_name);
      m_stream.Write(" "+property_name+" { get { return m_"+ field_name +"; } }\n");
    }

    public virtual void VisitExtended(ExtendedOptionInfo opt)
    {
      string property_type = CSharpCodeGenerator.ToCSharpType(opt.Type);
      _WritePropertyAndField(opt,property_type);
    }

    public virtual void VisitEnumeration(EnumerationOptionInfo opt)
    {
      string property_type = CSharpCodeGenerator.ToCSharpType(opt.Type);
      _WritePropertyAndField(opt,property_type);
    }

    public virtual void VisitScript(ScriptOptionInfo info)
    {
      Console.WriteLine("NOT IMPLEMENTED VisitScript");
    }

    public virtual void VisitSimple(SimpleOptionInfo opt)
    {
      string property_type;
      if (opt.IsMulti)
        property_type = CSharpVariableTypeGenerator.ElementTypeName(opt.SimpleType);
      else
        property_type = CSharpVariableTypeGenerator.TypeName(opt);
      _WritePropertyAndField(opt,property_type);
    }

    void _WritePropertyAndField(Option opt,string property_type)
    {
      string field_type_name = _FieldTypeName(opt);
      string field_name = CSharpCodeGenerator.ToFuncName(opt.Name);
      _WriteFieldName(opt,field_type_name);
      string property_name = CSharpCodeGenerator.ToClassName(field_name);
      if (opt.IsMulti){
        //Console.WriteLine("NOT IMPLEMENTED SimpleOptionInfo accessor for multi");
        m_stream.Write("  public ");
        m_stream.Write(property_type+"[]");
        m_stream.Write(" "+property_name+" { get { return m_"+ field_name +".Values; } }\n");
      }
      else{
        m_stream.Write("  public ");
        m_stream.Write(property_type);
        m_stream.Write(" "+property_name+" { ");
        if (field_type_name!=property_type){
          m_stream.Write(" get { return m_"+ field_name +".value(); } }\n");
        }
        else{
          m_stream.Write("[Arcane.Compiler.Directives.CppMethodGenerationInfo(IsField=true)]");
          m_stream.Write(" get { return m_"+ field_name +"; } }\n");
        }
      }
    }

    public virtual void VisitServiceInstance(ServiceInstanceOptionInfo opt)
    {
      //string field_name = CSharpCodeGenerator.toFuncName(opt.Name);
      //string property_name = CSharpCodeGenerator.toClassName(field_name);
      string property_type = CSharpCodeGenerator.ToCSharpType(opt.Type);
      _WritePropertyAndField(opt,property_type);
    }


    private void _WriteFieldName(Option opt,string field_type)
    {
      string field_name = CSharpCodeGenerator.ToFuncName(opt.Name);
      m_stream.Write("\n  private ");
      m_stream.Write(field_type);
      m_stream.Write("  m_"+field_name+";\n");
    }
    private string _FieldTypeName(Option opt)
    {
      //opt.accept(m_type_generator);
      return CSharpVariableTypeGenerator.TypeName(opt);
    }
  };

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * Classe implémentant le Design Pattern du visiteur pour générer le code C++ de
   * déclaration des classes pour les options de type complex.
   */
  public class CSharpClassDefinitionGenerator : IOptionInfoVisitor
  {
    public CSharpClassDefinitionGenerator(TextWriter stream)
    {
      m_stream = stream;
    }

    public virtual void VisitComplex(ComplexOptionInfo info)
    {
      foreach(Option o in info.Options){
        o.Accept(this);
      }

      if (info.Type == null)
        return;

      if (info.IsMulti) {
        string c_class_name;
        c_class_name = "CaseOption" + info.Type;
        string v_class_name;
        v_class_name = c_class_name + "Value";
        m_stream.Write("    public class " + v_class_name + "\n");
        m_stream.Write("    {\n");
        //m_stream.Write("    typedef " + v_class_name + " ThatClass;\n");
        m_stream.Write("    public " + v_class_name + "(Arcane.ICaseMng cm,");
        m_stream.Write("Arcane.ICaseOptionList icl,Arcane.XmlNode element)");
        m_stream.Write("{\n");
        m_stream.Write("  m_element = element;\n");

        CSharpOptionBuilderGenerator bg = new CSharpOptionBuilderGenerator(m_stream, "icl", "m_element");
        info.AcceptChildren(bg);

        CSharpBuilderBodyGenerator bbg = new CSharpBuilderBodyGenerator(m_stream);
        info.AcceptChildren(bbg);

        m_stream.Write("    }\n");
        m_stream.Write("     public Arcane.XmlNode Element { get { return m_element; } }\n");
        m_stream.Write("     private Arcane.XmlNode m_element;\n");

        CSharpVariableDefinitionGenerator vdg = new CSharpVariableDefinitionGenerator(m_stream);
        foreach( Option opt in info.Options){
          opt.Accept(vdg);
        }
        //info.AcceptChildren(vdg);

        m_stream.Write("    };\n");

        m_stream.Write("[Arcane.Compiler.Directives.CppClassGenerationInfo(TypeMapping=Arcane.Compiler.Directives.TypeMapping.ValueRef)]");
m_stream.Write("  public class " + c_class_name + "\n");
        m_stream.Write("  : Arcane.CaseOptionsMulti\n");
        //m_stream.Write(v_class_name + "*>\n");
        m_stream.Write("  {\n");
        m_stream.Write("  private "+v_class_name+"[] m_values;\n");
        m_stream.Write("\n");
        m_stream.Write("    public " + c_class_name);
        m_stream.Write("(Arcane.ICaseMng cm,Arcane.ICaseOptionList icl,string s,");
        m_stream.Write("Arcane.XmlNode element)\n");
        m_stream.Write("    : base(cm,icl,s,element," + info.MinOccurs
                       + "," + info.MaxOccurs + ") {\n");
        CSharpCodeGenerator.writeNameTranslations(m_stream,
                                             info.m_alternative_names,
                                             "");
        m_stream.Write("}\n");
        m_stream.Write("    public override void multiAllocate(Arcane.XmlNodeList elist){\n");
        m_stream.Write("      Integer s = elist.size();\n");
        m_stream.Write("      if (s!=0)\n");
        m_stream.Write("        m_values = new " + v_class_name + "[s];\n");
        m_stream.Write("      for( int i=0; i<s; ++i ){\n");
        m_stream.Write("        m_values[i] = new " + v_class_name);
        m_stream.Write(" (caseMng(),configList(),elist.node(i));\n");
        m_stream.Write("      }\n");
        m_stream.Write("    }\n");
        m_stream.Write("    public " + v_class_name + " this[Integer i]\n");
        m_stream.Write("    { get { return m_values[(int)i]; } }\n");
        m_stream.Write("    public Integer Length { get { return m_values.Length; } }\n");
        m_stream.Write("  };\n");
        m_stream.Write('\n');
      }
      else {
        string c_class_name;
        c_class_name = "CaseOption" + info.Type;
        m_stream.Write("  public class " + c_class_name);
        m_stream.Write("  : Arcane.CaseOptions");
        m_stream.Write("  {\n");
        m_stream.Write("    public " + c_class_name);
        m_stream.Write("(Arcane.ICaseMng cm,Arcane.ICaseOptionList icl,string s,");
        m_stream.Write("Arcane.XmlNode element)\n");
        m_stream.Write("    : base(icl,s,element)\n");
        m_stream.Write("    {\n");
        CSharpOptionBuilderGenerator bg = new CSharpOptionBuilderGenerator(m_stream, "configList()", "new Arcane.XmlNode()");
        info.AcceptChildren(bg);
        CSharpCodeGenerator.writeNameTranslations(m_stream,
                                             info.m_alternative_names,
                                             "");
        CSharpBuilderBodyGenerator bbg = new CSharpBuilderBodyGenerator(m_stream);
        info.AcceptChildren(bbg);

        m_stream.Write("}\n");
        //m_stream.Write("    const " + c_class_name + "& operator()() const { return *this; }\n");
        CSharpVariableDefinitionGenerator vdg = new CSharpVariableDefinitionGenerator(m_stream);
        info.AcceptChildren(vdg);
        m_stream.Write("  };\n");
        m_stream.Write('\n');
      }
    }

    public virtual void VisitExtended(ExtendedOptionInfo info) { }
    public virtual void VisitEnumeration(EnumerationOptionInfo info) { }
    public virtual void VisitScript(ScriptOptionInfo info) { }
    public virtual void VisitSimple(SimpleOptionInfo info) { }
    public virtual void VisitServiceInstance(ServiceInstanceOptionInfo info) { }

    private TextWriter m_stream;
  };

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
}
