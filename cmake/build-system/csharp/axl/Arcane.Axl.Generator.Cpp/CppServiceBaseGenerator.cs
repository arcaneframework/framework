using System;
using System.IO;
using System.Collections;
using Integer = System.Int32;

namespace Arcane.Axl
{
  /**
   * Classe générant le code de la classe de base d'un service. 
   */
  public class CppServiceBaseGenerator : CppCodeGenerator
  {
    /** Objet stockant les informations de l'élément XML "service". */
    private ServiceInfo m_info;
    /** Générateur des fichiers "CaseOptions.h" et "CaseOptions.cc". */
    private CppCaseOptionsGenerator m_co_generator;
    /** Nom du fichier xml contenant la description des options. */
    private string m_case_options_xml_file_name;

    public CppServiceBaseGenerator(string path,
                                string output_path,
                                ServiceInfo info)
      : base(path, output_path)
    {
      m_info = info;
      m_class_name = "Arcane" + m_info.Name + "Object";
      m_co_generator = new CppCaseOptionsGenerator(path,
                                                m_output_path,
                                                m_info.Name,
                                                m_info.NamespaceMacroName,
                                                m_info.NamespaceName,
                                                m_info.Version,
                                                m_info.Options,
                                                m_info.AlternativeNames,
                                                m_info.NotCaseOption);
      m_case_options_xml_file_name = m_info.Name;
      m_case_options_xml_file_name += "_";
      m_case_options_xml_file_name += path.Replace("/","_");
    }

    private string _serviceTypeStr(ServiceType t)
    {
      switch (t) {
        case ServiceType.ST_Application: return "Application";
        case ServiceType.ST_Session: return "Session";
        case ServiceType.ST_SubDomain: return "SubDomain";
        case ServiceType.ST_CaseOption: return "CaseOption";
        case ServiceType.ST_Unknown: return "Unknown";
      }
      return "Unknown";
    }

    public override void writeFile()
    {
      double version = m_info.Version;

      _writeClassFile(m_info.Name + "_axl.h", null, false);

      // generation du fichier CaseOptions
      if (m_co_generator!=null)
        m_co_generator.writeFile();

      TextWriter file_stream = new StringWriter();
      WriteInfo(file_stream);
      //string class_name_upper_case = m_class_name.clone();
      string class_name_upper_case = m_class_name.ToUpper();
      file_stream.Write("#ifndef ARCANE_" +class_name_upper_case + "_H\n");
      file_stream.Write("#define ARCANE_" + class_name_upper_case + "_H\n");
      WriteComments(file_stream);

      file_stream.Write("#include \"arcane/VariableTypes.h\"\n");
      file_stream.Write("#include \"arcane/ServiceInfo.h\"\n");
      file_stream.Write("#include \"arcane/ISubDomain.h\"\n");
      file_stream.Write("#include \"arcane/ServiceBuildInfo.h\"\n");
      file_stream.Write("#include \"arcane/ServiceRegisterer.h\"\n");
      file_stream.Write("#include \"arcane/ServiceFactory.h\"\n");
      file_stream.Write("#include \"arcane/ServiceFactory.h\"\n");
      file_stream.Write("#include \"arcane/BasicService.h\"\n");
      if (HasTests) file_stream.Write("#include \"arcane/UnitTestServiceAdapter.h\"\n");
			
      bool has_namespace = m_info.NamespaceMacroName != null || m_info.NamespaceName != null;

      if (version < 1.0) {
        WriteComments(file_stream);
        file_stream.Write("ARCANE_BEGIN_NAMESPACE\n");
        WriteComments(file_stream);
      }

      if (has_namespace) {
        WriteComments(file_stream);
        if (m_info.NamespaceMacroName != null)
          file_stream.Write(m_info.NamespaceMacroName + "_BEGIN_NAMESPACE\n");
        else
          file_stream.Write("namespace " + m_info.NamespaceName + "{\n");
      }

      if (m_co_generator!=null) {
        file_stream.Write("class " + m_co_generator.getClassName() + ";\n");
      }
      WriteComments(file_stream);


      file_stream.Write("//! Generation de la classe de base du service\n");
      file_stream.Write("class " + m_class_name + "\n");
      file_stream.Write(": public " + CppUtils.ConvertType(m_info.ParentName) + "\n");
      if (HasTests)
        file_stream.Write(", public Arcane::UnitTestServiceAdapter< " + m_class_name + " >\n");
      foreach(ServiceInfo.Interface itf in m_info.Interfaces){
        if (itf.IsInherited)
          file_stream.Write(", public " + CppUtils.ConvertType(itf.Name) + "\n");
      }
      file_stream.Write("{\n");
#if false
      // Generation de serviceInfoCreateFunction (obsolete)
      {
        file_stream.Write (" public:\n");
        file_stream.Write ("  static Arcane::ServiceInfo* serviceInfoCreateFunction(const Arcane::String& name)\n");
        file_stream.Write ("  {\n");
        file_stream.Write ("    Arcane::ServiceInfo* si = new Arcane::ServiceInfo(name,Arcane::VersionInfo(\"0.0\"),\n");
        file_stream.Write ("                                       Arcane::IServiceInfo::Dim2|Arcane::IServiceInfo::Dim3);\n");
        file_stream.Write ("    si->setCaseOptionsFileName(\"" + m_case_options_xml_file_name + "\");\n");
        //for (Integer i = 0, iss = m_info.m_interfaces.Count; i < iss; ++i) {
        foreach (ServiceInfo.Interface itf in m_info.Interfaces) {
          file_stream.Write ("    si->addImplementedInterface(\"" + itf.Name + "\");\n");
        }
        file_stream.Write ("    si->setAxlVersion(" + m_info.Version + ");\n");
        {
          string service_lower_case_name = Utils.ToLowerWithDash (m_info.Name);
          file_stream.Write ("    si->setDefaultTagName(Arcane::String(\""
          + service_lower_case_name + "\"));\n");
          foreach (DictionaryEntry de in m_info.AlternativeNames.m_names) {
            string key = (string)de.Key;
            string value = (string)de.Value;
            file_stream.Write ("    si->setTagName(Arcane::String(\""
            + key + "\"),Arcane::String(\"" + value + "\"));\n");
          }
        }
        file_stream.Write ("    return si;\n");
        file_stream.Write ("  }\n\n");
      }
#endif

      // Generation de fillServiceInfo()
      {
        file_stream.Write (" public:\n");
        file_stream.Write ("  template <typename ServiceClassType> static void fillServiceInfo(Arcane::ServiceInfo* si)\n");
        file_stream.Write ("  {\n");
        file_stream.Write ("    si->setCaseOptionsFileName(\"" + m_case_options_xml_file_name + "\");\n");
        foreach (ServiceInfo.Interface itf in m_info.Interfaces) {
          file_stream.Write ("  ARCANE_SERVICE_INTERFACE({0}) . registerToServiceInfo< ServiceClassType >(si);\n",itf.Name);
        }
        file_stream.Write ("    si->setAxlVersion(" + m_info.Version + ");\n");
        {
          string service_lower_case_name = Utils.ToLowerWithDash (m_info.Name);
          file_stream.Write ("    si->setDefaultTagName(Arcane::String(\""
          + service_lower_case_name + "\"));\n");
          foreach (DictionaryEntry de in m_info.AlternativeNames.m_names) {
            string key = (string)de.Key;
            string value = (string)de.Value;
            file_stream.Write ("    si->setTagName(Arcane::String(\""
            + key + "\"),Arcane::String(\"" + value + "\"));\n");
          }
        }
        file_stream.Write ("  }\n\n");
      }
      file_stream.Write(" public:\n");
      // Constructeur
      {
        file_stream.Write("  " + m_class_name + "(const Arcane::ServiceBuildInfo& sbi)\n");
        file_stream.Write("  : " + CppUtils.ConvertType(m_info.ParentName) + "(sbi)\n");
        if (HasTests)
          file_stream.Write("  , Arcane::UnitTestServiceAdapter< " + m_class_name + " >(this)\n");
        if (m_co_generator!=null) {
          file_stream.Write("  , m_options(0)\n");
        }

        _WriteVariablesConstructor(m_info.VariableInfoList,"sbi.mesh()",file_stream);

        file_stream.Write("  {\n");
        if (m_co_generator!=null) {
          string co_file_name = m_path;
          co_file_name += "_";
          co_file_name += m_info.Name;

          //file_stream.Write("  sbi.serviceInfo()->setCaseOptionsFileName(\""
          //            << m_case_options_xml_file_name << "\");\n";
          if (m_info.NotCaseOption) {
            file_stream.Write("    m_options = new " + m_co_generator.getClassName()
                              + "(sbi.subDomain()->caseMng());\n");
            //file_stream.Write("  m_options->setCaseService(this);\n";
          }
          else {
            file_stream.Write("    Arcane::ICaseOptions* co = sbi.caseOptions();\n");
            file_stream.Write("    if (co){\n");
            file_stream.Write("      m_options = new " + m_co_generator.getClassName() + "(sbi.subDomain()->caseMng(),co);\n");
            //file_stream.Write("      m_options->setCaseService(this);\n";
            file_stream.Write("    }\n");
          }
        }
				
        if (HasTests) {
          if (!String.IsNullOrEmpty(m_info.m_tests_info.m_class_set_up)) 
            file_stream.Write("    setClassSetUpFunction(&" + m_class_name + "::" + m_info.m_tests_info.m_class_set_up + ");\n");
          if (!String.IsNullOrEmpty(m_info.m_tests_info.m_class_tear_down)) 
            file_stream.Write("    setClassTearDownFunction(&" + m_class_name + "::" + m_info.m_tests_info.m_class_tear_down + ");\n");
          if (!String.IsNullOrEmpty(m_info.m_tests_info.m_test_set_up)) 
            file_stream.Write("    setTestSetUpFunction(&" + m_class_name + "::" + m_info.m_tests_info.m_test_set_up + ");\n");
          if (!String.IsNullOrEmpty(m_info.m_tests_info.m_test_tear_down)) 
            file_stream.Write("    setTestTearDownFunction(&" + m_class_name + "::" + m_info.m_tests_info.m_test_tear_down + ");\n");
          foreach(TestInfo ti in m_info.m_tests_info.m_test_info_list) 
            file_stream.Write("    addTestFunction(&" + m_class_name + "::" + ti.m_method_name + ", \"" + ti.m_name + "\", \"" + ti.m_method_name + "\");\n");
        }
				
        file_stream.Write("  }\n\n");
      }

      // Destructeur
      {
        file_stream.Write("  ~" + m_class_name + "()\n");
        file_stream.Write("  {\n");
        if (m_co_generator!=null) file_stream.Write("    delete m_options;\n");
        file_stream.Write("  }\n\n");
      }

      // Test comme méthodes abstraites
      if (HasTests) {
        file_stream.Write("  //! Méthodes de test\n");
        if (!String.IsNullOrEmpty(m_info.m_tests_info.m_class_set_up))
          file_stream.Write("  virtual void " + m_info.m_tests_info.m_class_set_up + "() = 0;\n");
        if (!String.IsNullOrEmpty(m_info.m_tests_info.m_class_tear_down))
          file_stream.Write("  virtual void " + m_info.m_tests_info.m_class_tear_down + "() = 0;\n");
        if (!String.IsNullOrEmpty(m_info.m_tests_info.m_test_set_up))
          file_stream.Write("  virtual void " + m_info.m_tests_info.m_test_set_up + "() = 0;\n");
        if (!String.IsNullOrEmpty(m_info.m_tests_info.m_test_tear_down))
          file_stream.Write("  virtual void " + m_info.m_tests_info.m_test_tear_down + "() = 0;\n");
        foreach(TestInfo ti in m_info.m_tests_info.m_test_info_list)
          file_stream.Write("  virtual void " + ti.m_method_name + "() = 0;  //!< " + ti.m_name + "\n");
        file_stream.Write("\n");
      }

      if (m_co_generator!=null) {
        file_stream.Write(" public:\n");
        file_stream.Write("  //! Options du jeu de données du service\n");
        file_stream.Write("  " + m_co_generator.getClassName() + "* options() const { return m_options; }\n");
        file_stream.Write("\n");
        file_stream.Write(" private:\n");
        file_stream.Write("  //! Options du jeu de données du service\n");
        file_stream.Write("  " + m_co_generator.getClassName() + "* m_options;\n");
        file_stream.Write("\n");
      }

      _WriteVariablesDeclaration(m_info.VariableInfoList,file_stream);
      file_stream.Write("};\n");
      file_stream.Write("\n");

      // ANCIENNE MACRO
      #if false
      {
        string service_upper_case = m_info.Name.ToUpper();
        string service_type_name = _serviceTypeStr(m_info.ServiceType);

        file_stream.Write("#define OLD_ARCANE_REGISTER_SERVICE_" + service_upper_case);
        file_stream.Write("(service_name,class_name,...) \\\n");
        file_stream.Write("extern \"C++\" ARCANE_EXPORT IServiceInfo* \\\n");
        file_stream.Write("arcaneCreateServiceInfo##class_name##service_name(const Arcane::String& name)\\\n");
        file_stream.Write("{ \\\n");
        file_stream.Write("  ServiceInfo* si = " + m_class_name + "::serviceInfoCreateFunction(name);\\\n");
        foreach( ServiceInfo.Interface itf in m_info.Interfaces){
          file_stream.Write("    si->addFactory(new " + service_type_name + "ServiceFactory2T< class_name,"
                            + CppUtils.ConvertType(itf.Name) + " >(si));\\\n");
        }
        file_stream.Write("  ServiceFactoryInfo* sfi = new ServiceFactoryInfo(si);\\\n");
        file_stream.Write("  si->setFactoryInfo(sfi);\\\n");
        file_stream.Write("  sfi->initProperties(__VA_ARGS__);\\\n");
        file_stream.Write("  return si;\\\n");
        file_stream.Write("} \\\n");
        file_stream.Write("ServiceRegisterer ARCANE_EXPORT globalServiceRegisterer##class_name##service_name");
        file_stream.Write("(arcaneCreateServiceInfo##class_name##service_name,#service_name)\n");
      }
      #endif

      // NOUVELLE MACRO
      {
        string service_upper_case = m_info.Name.ToUpper();
        string service_type_name = _serviceTypeStr(m_info.ServiceType);

        file_stream.Write("#define ARCANE_REGISTER_SERVICE_" + service_upper_case);
        file_stream.Write("(service_name,class_name) \\\n");
        file_stream.Write("  ARCANE_REGISTER_AXL_SERVICE(class_name,Arcane::ServiceProperty(#service_name,Arcane::ST_" +service_type_name+"))\n");
      }

      WriteComments(file_stream);
      if (has_namespace) {
        if (m_info.NamespaceMacroName != null)
          file_stream.Write(m_info.NamespaceMacroName + "_}\n");
        else
          file_stream.Write("}\n");
      }
      if (version < 1.0) {
        file_stream.Write("ARCANE_}\n");
      }
      WriteComments(file_stream);
      file_stream.Write("#endif\n");

      // write file
      _writeClassFile(m_info.Name + "_axl.h", file_stream.ToString(), true);
    }
		
    private bool HasTests { get { return m_info.m_tests_info != null && m_info.m_tests_info.m_test_info_list.Count > 0; } }
  }
}
