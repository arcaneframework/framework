// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseMng.cc                                                  (C) 2000-2025 */
/*                                                                           */
/* Classe gérant les options du jeu de données.                              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/List.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/Deleter.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/ApplicationInfo.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/CriticalSection.h"
#include "arcane/utils/CommandLineArguments.h"
#include "arcane/utils/internal/ParameterListWithCaseOption.h"

#include "arcane/core/ISubDomain.h"
#include "arcane/core/IApplication.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/IParallelSuperMng.h"
#include "arcane/core/ICaseMng.h"
#include "arcane/core/IModule.h"
#include "arcane/core/CaseOptions.h"
#include "arcane/core/XmlNode.h"
#include "arcane/core/XmlNodeList.h"
#include "arcane/core/XmlNodeIterator.h"
#include "arcane/core/ICaseDocument.h"
#include "arcane/core/ICaseFunctionProvider.h"
#include "arcane/core/CaseNodeNames.h"
#include "arcane/core/ISession.h"
#include "arcane/core/CaseTable.h"
#include "arcane/core/IMainFactory.h"
#include "arcane/core/IIOMng.h"
#include "arcane/core/ServiceFinder2.h"
#include "arcane/core/ObservablePool.h"
#include "arcane/core/ICaseDocumentVisitor.h"
#include "arcane/core/ServiceBuilder.h"
#include "arcane/core/internal/ICaseMngInternal.h"

#include "arcane/impl/CaseDocumentLangTranslator.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" std::unique_ptr<ICaseDocumentVisitor>
createPrintCaseDocumentVisitor(ITraceMng* tm,const String& lang);

extern "C++" ICaseDocumentFragment*
arcaneCreateCaseDocumentFragment(ITraceMng* tm,IXmlDocumentHolder* document);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestionnaire d'un cas.
 */
class CaseMng
: private ReferenceCounterImpl
, public ICaseMng
, public ICaseMngInternal
, public TraceAccessor
{
  ARCCORE_DEFINE_REFERENCE_COUNTED_INCLASS_METHODS();

 private:

  class OptionsReader
  {
   public:
    OptionsReader(ICaseMng*) {}
    void read(bool is_phase1);
    void addOption(ICaseOptions* o) { m_options.add(o); }
   private:
    UniqueArray<ICaseOptions*> m_options;
    void _read(bool is_phase1);
    void _searchInvalidOptions();
  };
 private:

  class ErrorInfo
  {
   public:
    explicit ErrorInfo(const String& error_message)
    : m_has_error(true), m_error_message(error_message){}
    ErrorInfo() : m_has_error(false){}
   public:
    bool hasError() const { return m_has_error; }
    const String& errorMessage() const { return m_error_message; }
   private:
    bool m_has_error;
    String m_error_message;
  };

  /*!
   * Classe pour filtrer les options et ne garder que celles
   * dont les modules sont utilisés.
   */
  class CaseOptionsFilterUsed
  {
   public:
    explicit CaseOptionsFilterUsed(const CaseOptionsList& opt_list)
    : m_begin(opt_list.begin()), m_end(opt_list.end()){}
   public:
    auto begin() const { return m_begin; }
    auto end() const { return m_end; }
    void operator++()
    {
      bool do_continue = true;
      do {
        ++m_begin;
        if (m_begin >= m_end)
          return;
        ICaseOptions* co = *m_begin;
        IModule* md = co->caseModule();
        do_continue = (md && !md->used());
      } while (do_continue);
    }
   private:
    CaseOptionsList::const_iterator m_begin;
    CaseOptionsList::const_iterator m_end;
  };

 public:

  explicit CaseMng(ISubDomain*);
  ~CaseMng() override;

 public:
	
  ITraceMng* traceMng() override { return TraceAccessor::traceMng(); }
  IMeshMng* meshMng() const override { return m_sub_domain->meshMng(); }
  ISubDomain* subDomain() override { return m_sub_domain; }
  IApplication* application() override { return m_sub_domain->application(); }
  IPhysicalUnitSystem* physicalUnitSystem() const override { return m_sub_domain->physicalUnitSystem(); }
  ICaseDocument* caseDocument() override { return m_case_document.get(); }
  ICaseDocumentFragment* caseDocumentFragment() override
  {
    ICaseDocument* doc = caseDocument();
    if (doc)
      return doc->fragment();
    return nullptr;
  }
  ICaseDocument* readCaseDocument(const String& filename,ByteConstArrayView bytes) override;

  void readFunctions() override;
  void readOptions(bool is_phase1) override;
  void printOptions() override;
  void registerOptions(ICaseOptions*) override;
  void unregisterOptions(ICaseOptions*) override;
  ICaseFunction* findFunction(const String& name) const override;
  void updateOptions(Real current_time,Real current_deltat,Integer current_iteration) override;
  CaseFunctionCollection functions() override { return m_functions; }

  void removeFunction(ICaseFunction* func,bool dofree) override;
  void removeFunction(ICaseFunction* func) override;
  void addFunction(ICaseFunction* func) override
  {
    addFunction(makeRef(func));
  }
  void addFunction(Ref<ICaseFunction> func) override;

  CaseOptionsCollection blocks() const override { return m_case_options_list; }

  void setTreatWarningAsError(bool v) override { m_treat_warning_as_error = v; }
  bool isTreatWarningAsError() const override { return m_treat_warning_as_error; }

  void setAllowUnkownRootElelement(bool v) override { m_allow_unknown_root_element = v; }
  bool isAllowUnkownRootElelement() const override { return m_allow_unknown_root_element; }

  IObservable* observable(eCaseMngEventType type) override
  {
    return m_observables[type];
  }

  Ref<ICaseFunction> findFunctionRef(const String& name) const;

  Ref<ICaseMng> toReference() override { return makeRef<ICaseMng>(this); }

  ICaseMngInternal* _internalImpl() override { return this; }

  //! Implémentation via ICaseMngInternal
  //@{
  void internalReadOneOption(ICaseOptions* opt, bool is_phase1) override;
  ICaseDocumentFragment* createDocumentFragment(IXmlDocumentHolder* document) override
  {
    return arcaneCreateCaseDocumentFragment(traceMng(),document);
  }
  const ParameterListWithCaseOption& parameters() const { return  m_parameters; }
  //@}
  
 public:
	
  String msgClassName() const { return "CaseMng"; }

 private:

  ISubDomain* m_sub_domain; //!< Gestionnaire de sous-domain
  ScopedPtrT<ICaseDocument> m_case_document;
  CaseFunctionList m_functions; //!< Liste des fonctions
  CaseOptionsList m_case_options_list; //!< Liste des options du cas
  List<CaseOptionBase*> m_options_with_function;
  bool m_treat_warning_as_error = false;
  bool m_allow_unknown_root_element = true;
  ObservablePool<eCaseMngEventType> m_observables;
  //! Indique si les fonctions ont déjà été lues
  bool m_is_function_read = false;
  Ref<ICaseFunctionDotNetProvider> m_dotnet_provider;
  ParameterListWithCaseOption m_parameters;

 private:

  ErrorInfo _readOneTable(const XmlNode& func_elem);
  ErrorInfo _checkValidFunction(const XmlNode& func_elem,CaseFunctionBuildInfo& cfbi);
  void _readOptions(bool is_phase1);
  void _readFunctions();
  void _readCaseDocument(const String& filename,ByteConstArrayView bytes);
  void _printErrors(ICaseDocumentFragment* doc,bool is_phase1);
  void _checkTranslateDocument();
  void _removeFunction(ICaseFunction* func,bool do_delete);
  void _searchInvalidOptions();
  ICaseDocument* _noNullCaseDocument()
  {
    ICaseDocument* doc = m_case_document.get();
    ARCANE_CHECK_POINTER(doc);
    return doc;
  }
  void _readOneDotNetFunction(const String& assembly_name,const String& class_name);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseMng::OptionsReader::
read(bool is_phase1)
{
  _read(is_phase1);
  if (is_phase1)
    _searchInvalidOptions();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseMng::OptionsReader::
_read(bool is_phase1)
{
  auto read_phase = (is_phase1) ? eCaseOptionReadPhase::Phase1 : eCaseOptionReadPhase::Phase2;
  for( ICaseOptions* co : m_options){
    co->read(read_phase);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseMng::OptionsReader::
_searchInvalidOptions()
{
  // Cherche les éléments du jeu de données qui ne correspondent pas
  // à une option connue.
  XmlNodeList invalid_elems;
  for( ICaseOptions* co : m_options)
    co->addInvalidChildren(invalid_elems);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseMng::
CaseMng(ISubDomain* sub_domain)
: TraceAccessor(sub_domain->traceMng())
, m_sub_domain(sub_domain)
, m_treat_warning_as_error(false)
{
  if (!platform::getEnvironmentVariable("ARCANE_STRICT_CASEOPTION").null())
    m_treat_warning_as_error = true;

  m_observables.add(eCaseMngEventType::BeginReadOptionsPhase1);
  m_observables.add(eCaseMngEventType::BeginReadOptionsPhase2);

  // Recopie dans \a m_parameters les arguments de la ligne de commande
  // qui pourraient surcharger les valeurs du jeu de données
  m_parameters.addParameters(sub_domain->application()->applicationInfo().commandLineArguments().parameters());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseMng::
~CaseMng()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ICaseMng*
arcaneCreateCaseMng(ISubDomain* mng)
{
  return new CaseMng(mng);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseMng::
readOptions(bool is_phase1)
{
  Trace::Setter mci(traceMng(),msgClassName());

  ICaseDocumentFragment* doc = _noNullCaseDocument()->fragment();
  if (is_phase1)
    info() << "Reading the input data (phase1): language '"
           << doc->language() << "', "
           << m_case_options_list.count() << " input data.";
  else
    info() << "Reading the input data (phase2)";

  if (is_phase1){
    doc->clearErrorsAndWarnings();
  }

  if (is_phase1)
    readFunctions();

  // Notifie du début de lecture des options.
  if (is_phase1)
    m_observables[eCaseMngEventType::BeginReadOptionsPhase1]->notifyAllObservers();
  else
    m_observables[eCaseMngEventType::BeginReadOptionsPhase2]->notifyAllObservers();

  _readOptions(is_phase1);
  _printErrors(doc,is_phase1);

  _checkTranslateDocument();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseMng::
_checkTranslateDocument()
{
  // Si demandé, écrit un fichier contenant la traduction dans le langage spécifié
  // de chaque élément du jeu de données.
  String tr_lang = platform::getEnvironmentVariable("ARCANE_TRANSLATE_CASEDOCUMENT");
  if (!tr_lang.null()){
    info() << "Generating translation for case file to lang=" << tr_lang;
    CaseDocumentLangTranslator translator(traceMng());
    String convert_string = translator.translate(this,tr_lang);
    {
      std::ofstream ofile("convert_info.txt");
      convert_string.writeBytes(ofile);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseMng::
_printErrors(ICaseDocumentFragment* doc,bool is_phase1)
{
  //ICaseDocumentFragment* doc = _noNullCaseDocument()->fragment();

  // Affiche les avertissements mais uniquement lors de la phase2 pour les avoir
  // tous en une fois (certains avertissements ne sont générés que lors de la phase2)
  if (!m_treat_warning_as_error){
    if (!is_phase1){
      if (doc->hasWarnings()){
        OStringStream ostr;
        doc->printWarnings(ostr());
        pwarning() << "The input data are not coherent:\n\n" << ostr.str();
      }
    }
  }
  bool has_error = doc->hasError();
  if (doc->hasWarnings() && m_treat_warning_as_error && !is_phase1)
    has_error = true;

  if (has_error){
    OStringStream ostr;
    if (m_treat_warning_as_error && doc->hasWarnings())
      doc->printWarnings(ostr());
    doc->printErrors(ostr());
    ARCANE_FATAL("Input data are invalid:\n\n{0}",ostr.str());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseMng::
readFunctions()
{
  if (m_is_function_read)
    return;

  Trace::Setter mci(traceMng(),msgClassName());

  // Enregistre les services fournissant les fonctions
  {
    typedef ServiceFinder2T<ICaseFunctionProvider,ISubDomain> FinderType;
    ISubDomain* sd = subDomain();
    FinderType finder(sd->application(),sd);
    const Array<FinderType::FactoryType*>& factories = finder.factories();
    info() << "NB_CASE_FUNCTION_PROVIDER_FACTORY = " << factories.size();
    for( const auto& factory : factories ){
      auto cfp = factory->createServiceReference(ServiceBuildInfoBase(sd));
      if (cfp.get()){
        info() << "FOUND CASE FUNCTION PROVIDER (V2)!";
        cfp->registerCaseFunctions(this);
      }
    }
  }

  _readFunctions();

  m_is_function_read = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseMng::
registerOptions(ICaseOptions* co)
{
  Trace::Setter mci(traceMng(),msgClassName());

  log() << "Register case option <" << co->rootTagName() << ">";
  m_case_options_list.add(co);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseMng::
unregisterOptions(ICaseOptions* co)
{
  m_case_options_list.remove(co);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseMng::
_readFunctions()
{
  bool has_error = false;

  // Lecture des fonctions

  ICaseDocumentFragment* doc = _noNullCaseDocument()->fragment();
  XmlNode case_root_elem = doc->rootElement();

  CaseNodeNames* cnn = doc->caseNodeNames();

  // Récupère la liste des tables de marche.
  XmlNode funcs_elem = case_root_elem.child(cnn->functions);
  XmlNodeList functions_elem = funcs_elem.children();

  String ustr_table(cnn->function_table);
  String ustr_script(cnn->function_script);

  if (functions_elem.empty())
    return;
	
  for( XmlNode node : functions_elem ){
    bool is_bad = false;
    if (node.type()!=XmlNode::ELEMENT)
      continue;
    if (node.isNamed(ustr_table)){
      ErrorInfo err_info = _readOneTable(node);
      if (err_info.hasError()){
        is_bad = true;
        String table_name = node.attrValue(cnn->name_attribute);
        if (table_name.null())
          error() << " Error in element '" << node.xpathFullName() << "' : "
                  << err_info.errorMessage();
        else
          error() << " Error in table named '" << table_name << "' : "
                  << err_info.errorMessage();
      }
    }
    else if (node.isNamed("external-assembly")){
      String assembly_name = node.expectedChild("assembly-name").value();
      String class_name = node.expectedChild("class-name").value();
      _readOneDotNetFunction(assembly_name,class_name);
    }
    else
      warning() << "Unknown node in functions: " << node.xpathFullName();
    has_error |= is_bad;
  }

  if (has_error)
    ARCANE_FATAL("Error while reading the functions");

  // Affiche des informations sur le nombre de tables et leur nombre
  // d'éléments
  log() << "Number of functions: " << m_functions.count();
  for( auto& icf_ref : m_functions ){
    auto table = dynamic_cast<CaseTable*>(icf_ref.get());
    if (table)
      log() << "Table <" << table->name() << "> own "
            << table->nbElement() << " element(s)";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseMng::ErrorInfo CaseMng::
_checkValidFunction(const XmlNode& func_elem,CaseFunctionBuildInfo& cfbi)
{
  CaseNodeNames* cnn = caseDocument()->caseNodeNames();

  String func_name = func_elem.attrValue(cnn->name_attribute);
  if (func_name.null())
    return ErrorInfo(String::format("missing attribute '{0}'",cnn->name_attribute));

  cfbi.m_name = func_name;
  if (findFunction(func_name))
    return ErrorInfo(String::format("function '{0}' is defined several times",func_name));

  String param_name = func_elem.attrValue(cnn->function_parameter);
  if (param_name.null())
    return ErrorInfo(String::format("missing attribute '{0}'",cnn->function_parameter));

  String value_name = func_elem.attrValue(cnn->function_value);
  if (value_name.null())
    return ErrorInfo(String::format("missing attribute '{0}'",cnn->function_value));

  String comul_name = func_elem.attrValue("comul");
  String transform_x_name = func_elem.attrValue("comul-x");

  cfbi.m_transform_param_func = transform_x_name;
  cfbi.m_transform_value_func = comul_name;

  String ustr_time(cnn->time_type);
  String ustr_iteration(cnn->iteration_type);

  String ustr_real(cnn->real_type);
  String ustr_real3(cnn->real3_type);
  String ustr_bool(cnn->bool_type);
  String ustr_integer(cnn->integer_type);
  String ustr_string(cnn->string_type);

  UniqueArray<String> valid_param_strs = { ustr_time, ustr_real, ustr_iteration, ustr_integer };

  // Vérifie que le type de paramètre spécifié est correct (réel ou entier)
  ICaseFunction::eParamType param_type = ICaseFunction::ParamUnknown;
  if (param_name==ustr_time || param_name==ustr_real)
    param_type = ICaseFunction::ParamReal;
  else if (param_name==ustr_iteration || param_name==ustr_integer)
    param_type = ICaseFunction::ParamInteger;
  if (param_type==ICaseFunction::ParamUnknown){
    return ErrorInfo(String::format("invalid value '{0}' for attribute '{1}'. Valid values are '{2}'.",
                                    param_name,cnn->function_parameter,String::join(", ",valid_param_strs)));
  }
  cfbi.m_param_type = param_type;

  UniqueArray<String> valid_value_strs = { ustr_real, ustr_real3, ustr_integer, ustr_bool, ustr_string };

  // Vérifie que le type de valeur spécifié est correct (réel ou entier)
  ICaseFunction::eValueType value_type = ICaseFunction::ValueUnknown;
  if (value_name==ustr_real)
    value_type = ICaseFunction::ValueReal;
  if (value_name==ustr_integer)
    value_type = ICaseFunction::ValueInteger;
  if (value_name==ustr_bool)
    value_type = ICaseFunction::ValueBool;
  if (value_name==ustr_string)
    value_type = ICaseFunction::ValueString;
  if (value_name==ustr_real3)
    value_type = ICaseFunction::ValueReal3;
  if (value_type==ICaseFunction::ValueUnknown)
    return ErrorInfo(String::format("invalid value '{0}' for attribute '{1}'. Valid values are '{2}'.",
                                    value_name,cnn->function_value,String::join(", ",valid_value_strs)));
  cfbi.m_value_type = value_type;

  // Regarde s'il y a un élément 'deltat-coef' et si oui récupère sa valeur.
  String deltat_coef_str = func_elem.attrValue(cnn->function_deltat_coef);
  Real deltat_coef = 0.0;
  if (!deltat_coef_str.null()){
    if (builtInGetValue(deltat_coef,deltat_coef_str))
      return ErrorInfo(String::format("Invalid value '{0}' for attribute '{1}. Can not convert to 'Real' type",
                                      deltat_coef_str,cnn->function_deltat_coef));
    cfbi.m_deltat_coef = deltat_coef;
    info() << "Coefficient deltat for the function '" << func_name << "' = " << cfbi.m_deltat_coef;
  }

  return ErrorInfo();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseMng::ErrorInfo CaseMng::
_readOneTable(const XmlNode& func_elem)
{
  CaseFunctionBuildInfo cfbi(traceMng());

  ErrorInfo err_info = _checkValidFunction(func_elem,cfbi);
  if (err_info.hasError())
    return err_info;

  CaseNodeNames* cnn = caseDocument()->caseNodeNames();

  String ustr_constant(cnn->function_constant);
  String ustr_linear(cnn->function_linear);
  String ustr_value(cnn->function_value);

  String interpolation_name = func_elem.attrValue(cnn->function_interpolation);
  if (interpolation_name.null())
    return ErrorInfo(String::format("missing attribute '{0}'",cnn->function_interpolation));

  // Vérifie que le type de la courbe spécifié est correct
  CaseTable::eCurveType interpolation_type = CaseTable::CurveUnknown;
  if (interpolation_name==ustr_constant)
    interpolation_type = CaseTable::CurveConstant;
  else if (interpolation_name==ustr_linear)
    interpolation_type = CaseTable::CurveLinear;
  if (interpolation_type==CaseTable::CurveUnknown){
    return ErrorInfo(String::format("Invalid value for attribute '{0}'. Valid values are '{1}' or '{2}'",
                                    cnn->function_interpolation,ustr_constant,ustr_linear));
  }

  // Récupère la liste des valeurs de la fonction
  XmlNodeList func_value_list = func_elem.children(ustr_value);
  Integer nb_func_value = func_value_list.size();
  if (nb_func_value==0)
    return ErrorInfo("The table has no values");

  auto func = new CaseTable(cfbi,interpolation_type);
  m_functions.add(Ref<ICaseFunction>::create(func));

  Integer value_index = 0;
  
  String ustr_x("x");
  String ustr_y("y");
  for( const auto& i : func_value_list ) {
    String param_str = i.child(ustr_x).value();
    String value_str = i.child(ustr_y).value();
    param_str = String::collapseWhiteSpace(param_str);
    value_str = String::collapseWhiteSpace(value_str);
    if (param_str.null())
      return ErrorInfo(String::format("index={0} element <x> is missing",value_index));
    if (value_str.null())
      return ErrorInfo(String::format("index={0} element <y> is missing",value_index));

    CaseTable::eError error_number = func->appendElement(param_str,value_str);
		// TODO: AJOUTER DOC DANS LE JDD sur ces informations.
    if (error_number!=CaseTable::ErrNo){
      String message = "No info";
      switch(error_number){
      case CaseTable::ErrCanNotConvertParamToRightType:
        message = String::format("parameter '{0}' can not be converted to the 'param' type of the table",param_str);
        break;
      case CaseTable::ErrCanNotConvertValueToRightType:
        message = String::format("value '{0}' can not be converted to the 'value' type of the table",value_str);
        break;
      case CaseTable::ErrNotGreaterThanPrevious:
        message = "<x> lesser than previous <x>";
        break;
      case CaseTable::ErrNotLesserThanNext:
        message = "<x> greater than next <x>";
        break;
      case CaseTable::ErrBadRange:
        message = "bad interval";
        break;
      case CaseTable::ErrNo:
        // Ne devrait jamais arriver
        ARCANE_FATAL("Internal Error");
      }
      return ErrorInfo(String::format("index={0} : {1}",value_index,message));
    }
    ++value_index;
  }
  return ErrorInfo();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseMng::
_readOneDotNetFunction(const String& assembly_name,const String& class_name)
{
  // Si ce n'est pas encore fait, charge le service '.Net' qui enregistrera
  // les fonctions
  if (!m_dotnet_provider.get()){
    ServiceBuilder<ICaseFunctionDotNetProvider> sb(subDomain());
    m_dotnet_provider = sb.createReference("ArcaneDefaultDotNetCaseFunctionProvider",SB_AllowNull);
    if (!m_dotnet_provider.get())
      ARCANE_FATAL("Can not create '.Net' case function provider. Check that the '.Net' runtime is loaded.");
  }

  m_dotnet_provider->registerCaseFunctions(this,assembly_name,class_name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Recherche les options invalides du jeu de données.
 */
void CaseMng::
_searchInvalidOptions()
{
  ICaseDocument* doc = _noNullCaseDocument();
  ICaseDocumentFragment* doc_fragment = doc->fragment();
  // Cherche les éléments du jeu de données qui ne correspondent pas
  // à une option connue.
  XmlNodeList invalid_elems;
  for( ICaseOptions* co : CaseOptionsFilterUsed(m_case_options_list))
    co->addInvalidChildren(invalid_elems);

  // Cherche les éléments racines qui ne correspondent pas à une option

  // Temporairement pour des raisons de compatibilité, autorise les
  // éléments à la racine qui ne sont pas utilisés.
  if (!platform::getEnvironmentVariable("ARCANE_ALLOW_UNKNOWN_ROOT_ELEMENT").null())
    m_allow_unknown_root_element = true;

  String arcane_element_name = doc->arcaneElement().name();
  String function_element_name = doc->functionsElement().name();
  XmlNodeList mesh_elements = doc->meshElements();
  String mesh_element_name;
  if (mesh_elements.size()>0)
    mesh_element_name = mesh_elements[0].name();
  for( XmlNode node : doc_fragment->rootElement() ){
    if (node.type()!=XmlNode::ELEMENT)
      continue;
    String name = node.name();
    if (name=="comment")
      continue;
    if (name==arcane_element_name || name==function_element_name)
      continue;
    if (!mesh_element_name.null() && name==mesh_element_name)
      continue;
    bool is_found = false;
    for( ICaseOptions* co : CaseOptionsFilterUsed(m_case_options_list)){
      if (co->rootTagName()==name){
        is_found = true;
        break;
      }
    }
    if (!is_found){
      if (m_allow_unknown_root_element)
        pwarning() << "-- Unknown root option '" << node.xpathFullName() << "'";
      else
        invalid_elems.add(node);
    }
  }

  if (!doc_fragment->hasError()){
    if (!invalid_elems.empty()){
      for( XmlNode xnode : invalid_elems ){
        perror() << "-- Unknown root option '" << xnode.xpathFullName() << "'";
      }
      pfatal() << "Unknown root option(s) in the input data. "
               << "You can put these options inside <comment> tags to remove this error";
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseMng::
_readOptions(bool is_phase1)
{
  auto read_phase = (is_phase1) ? eCaseOptionReadPhase::Phase1 : eCaseOptionReadPhase::Phase2;
  for( ICaseOptions* co : CaseOptionsFilterUsed(m_case_options_list)){
    co->read(read_phase);
  }

  if (is_phase1){
    _searchInvalidOptions();
  }

  if (!is_phase1){
    // Recherche les options utilisant des fonctions.
    m_options_with_function.clear();
    UniqueArray<CaseOptionBase*> col;
    for( ICaseOptions* co : CaseOptionsFilterUsed(m_case_options_list)){
      co->deepGetChildren(col);
    }
    for( CaseOptionBase* co : col ){
      if (co->function()){
        m_options_with_function.add(co);
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseMng::
printOptions()
{
  String lang = _noNullCaseDocument()->fragment()->language();

  auto v = createPrintCaseDocumentVisitor(traceMng(),lang);
  info() << "-----------------------------------------------------";
  info();
  info() << "Input data values:";
  // Par défaut, utilise le mécanisme historique d'affichage pour que les
  // utilisateurs n'aient pas trop de différences d'affichage avec les
  // nouvelles versions de Arcane.
  // TODO: vérifier que le nouvel affichage est identique à l'ancien pour
  // la plupart des options.
  const bool use_old = true;
  for( ICaseOptions* co : CaseOptionsFilterUsed(m_case_options_list)){
    if (use_old)
      co->printChildren(lang,0);
    else
      co->visit(v.get());
  }

  info() << "-----------------------------------------------------";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseMng::
updateOptions(Real current_time,Real current_deltat,Integer current_iteration)
{
  Trace::Setter mci(traceMng(),msgClassName());

  for( CaseOptionBase* co : m_options_with_function ){
    ICaseFunction* cf = co->function();
    Real deltat_coef = cf->deltatCoef();
    Real t = current_time;
    if (!math::isZero(deltat_coef)){
      Real to_add = deltat_coef * current_deltat;
      t += to_add;
    }
    co->updateFromFunction(t,current_iteration);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<ICaseFunction> CaseMng::
findFunctionRef(const String& name) const
{
  for( auto& func_ref : m_functions )
    if (func_ref->name()==name)
      return func_ref;
  return {};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ICaseFunction* CaseMng::
findFunction(const String& name) const
{
  return findFunctionRef(name).get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseMng::
removeFunction(ICaseFunction* func)
{
  _removeFunction(func,false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseMng::
removeFunction(ICaseFunction* func,bool do_free)
{
  _removeFunction(func,do_free);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseMng::
_removeFunction(ICaseFunction* func,bool do_delete)
{
  // Plus utilisé depuis qu'on utilise un compte de référence.
  ARCANE_UNUSED(do_delete);
  Integer index = 0;
  bool is_found = false;
  for( auto& f : m_functions ){
    if (f.get()==func){
      is_found = true;
      break;
    }
    ++index;
  }
  if (is_found){
    m_functions.removeAt(index);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseMng::
addFunction(Ref<ICaseFunction> func)
{
  m_functions.add(func);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ICaseDocument* CaseMng::
readCaseDocument(const String& filename,ByteConstArrayView case_bytes)
{
  IParallelSuperMng* sm = application()->parallelSuperMng();
  {
    // Pour l'instant lit dans la section critique car cela provoque
    // certains plantages sinon de temps en temps (à étudier)
    CriticalSection cs(sm->threadMng());
    _readCaseDocument(filename,case_bytes);
  }
  return m_case_document.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseMng::
_readCaseDocument(const String& filename,ByteConstArrayView case_bytes)
{
  if (m_case_document.get())
    // Déjà lu...
    return;

  IApplication* app = m_sub_domain->application();
  IIOMng* io_mng = m_sub_domain->parallelMng()->ioMng();
  String printed_filename = filename;

  IXmlDocumentHolder* case_doc = io_mng->parseXmlBuffer(case_bytes,filename);
  if (!case_doc){
    info() << "XML Memory File:";
    info() << filename;
    pfatal() << "Failed to analyze the input data " << printed_filename;
  }

  m_case_document = app->mainFactory()->createCaseDocument(app,case_doc);

  CaseNodeNames* cnn = caseDocument()->caseNodeNames();

  String code_name = app->applicationInfo().codeName();
  XmlNode root_elem = m_case_document->fragment()->rootElement();
  if (root_elem.name()!=cnn->root){
    pfatal() << "The root element <" << root_elem.name() << "> has to be <" << cnn->root << ">";
  }

  if (root_elem.attrValue(cnn->code_name)!=code_name){
    pfatal() << "The file is not a case of code '" << code_name << "'";
  }

  String case_codeversion = root_elem.attrValue(cnn->code_version);
  String code_version(app->majorAndMinorVersionStr());
  String code_version2(app->mainVersionStr());
  if (case_codeversion!=code_version && case_codeversion!=code_version2){
    if (!m_sub_domain->session()->checkIsValidCaseVersion(case_codeversion))
      pfatal() << "The version number of the file (" << case_codeversion
              << ") doesn't match the version of the code '"
              << code_name << "' (" << code_version << ").";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseMng::
internalReadOneOption(ICaseOptions* opt,bool is_phase1)
{
  ARCANE_CHECK_POINTER(opt);
  ICaseDocumentFragment* doc = opt->caseDocumentFragment();
  ARCANE_CHECK_POINTER(doc);
  if (is_phase1)
    doc->clearErrorsAndWarnings();
  OptionsReader reader(this);
  reader.addOption(opt);
  reader.read(is_phase1);
  XmlNodeList invalid_elems;
  opt->addInvalidChildren(invalid_elems);
  _printErrors(doc,is_phase1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
