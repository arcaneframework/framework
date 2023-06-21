// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Hdf5VariableReader.cc                                       (C) 2000-2023 */
/*                                                                           */
/* Lecture de variables au format HDF5.                                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/HashTableMap.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/NotSupportedException.h"

#include "arcane/AbstractService.h"
#include "arcane/IVariableReader.h"
#include "arcane/BasicTimeLoopService.h"
#include "arcane/IXmlDocumentHolder.h"
#include "arcane/IIOMng.h"
#include "arcane/IMesh.h"
#include "arcane/IItemFamily.h"
#include "arcane/IParallelMng.h"
#include "arcane/ISubDomain.h"
#include "arcane/CommonVariables.h"
#include "arcane/IVariableAccessor.h"
#include "arcane/Directory.h"
#include "arcane/VariableCollection.h"
#include "arcane/IMeshMng.h"

#include "arcane/hdf5/Hdf5VariableReader_axl.h"
#include "arcane/hdf5/Hdf5Utils.h"
#include "arcane/hdf5/Hdf5VariableInfoBase.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Hdf5Utils;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class Hdf5VariableReaderHelperBase
: public TraceAccessor
{
 protected:

  class TimePathPair
  {
   public:
    TimePathPair()
    : m_time(0.0) {}
    TimePathPair(Real vtime,const String& path)
    : m_time(vtime), m_path(path) {}
   public:
    Real timeValue() const { return m_time; }
    const String& path() const { return m_path; }
   private:
    Real m_time;
    String m_path;
  };

 public:
  
  Hdf5VariableReaderHelperBase(IMesh* mesh)
  : TraceAccessor(mesh->traceMng()), m_mesh(mesh), m_is_verbose(false)
  {
    if (!platform::getEnvironmentVariable("ARCANE_DEBUG_HDF5VARIABLE").null())
      m_is_verbose = true;
  }

 protected:
  
  IMesh* m_mesh;
  Hdf5Utils::StandardTypes m_types;
  String m_hdf5_file_name;
  bool m_is_verbose;

 protected:

  void _readStandardArray(IVariable* var,RealArray& buffer,hid_t file_id,const String& path);
  void _readVariable(IVariable* var,RealArray& buffer,HFile& hfile,const String& path);
  void _checkValidVariable(IVariable* var);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Hdf5VariableReaderHelperBase::
_readStandardArray(IVariable* var,RealArray& buffer,hid_t file_id,const String& path)
{
  Hdf5Utils::StandardArrayT<Real> values(file_id,path);
  values.readDim();
  Int64ConstArrayView dims(values.dimensions());
  Integer nb_dim = dims.size();
  if (nb_dim!=1)
    fatal() << "Only one-dimension array are allowed "
            << " dim=" << nb_dim << " var_name=" << var->fullName() << " path=" << path;
  Integer nb_item = arcaneCheckArraySize(dims[0]);
  info(4) << "NB_ITEM: nb_item=" << nb_item;
#if 0
  if (nb_item!=family->nbItem())
    fatal() << "Bad number of items in file n=" << nb_item
            << " expected=" << family->nbItem()
            << " var_name=" << var_name << " path=" << var_path
            << " family=" << var_family;
#endif
  buffer.resize(nb_item);
  values.read(m_types,buffer);
#if 0
  {
    Integer index=0;
    for( Integer i=0; i<nb_item; ++i ){
      if (buffer[i]>0.0){
        ++index;
        info() << " VAR_VAL i=" << i << " value=" << buffer[i];
      }
      if (index>20)
        break;
    }
  }
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Hdf5VariableReaderHelperBase::
_readVariable(IVariable* var,RealArray& buffer,HFile& hfile,const String& path)
{
  IParallelMng* pm = m_mesh->parallelMng();
  bool is_master = pm->isMasterIO();
  //Integer rank = pm->commRank();
  Integer master_rank = pm->masterIORank();
  Integer buf_size = 0;

  if (is_master){
    if (hfile.id()<0){
      info() << "Hdf5VariableReaderHelper::OPEN FILE " << m_hdf5_file_name;
      hfile.openRead(m_hdf5_file_name);
    }
    _readStandardArray(var,buffer,hfile.id(),path);
    buf_size = buffer.size();
    IntegerArrayView iav(1,&buf_size);
    pm->broadcast(iav,master_rank);
    pm->broadcast(buffer,master_rank);
  }
  else{
    IntegerArrayView iav(1,&buf_size);
    pm->broadcast(iav,master_rank);
    buffer.resize(buf_size);
    pm->broadcast(buffer,master_rank);
  }
    
  Int64UniqueArray unique_ids(buf_size);
  Int32UniqueArray local_ids(buf_size);
  IData* var_data = var->data();
  auto* var_true_data = dynamic_cast< IArrayDataT<Real>* >(var_data);
  if (!var_true_data)
    throw FatalErrorException(A_FUNCINFO,"Variable is not an array of Real");
  RealArrayView var_value(var_true_data->view());
  Integer nb_var_value = var_value.size();
  for( Integer z=0; z<buf_size; ++z )
    unique_ids[z] = z;
  var->itemFamily()->itemsUniqueIdToLocalId(local_ids,unique_ids,false);
  for( Integer z=0; z<buf_size; ++z ){
    Integer lid = local_ids[z];
    if (lid==NULL_ITEM_LOCAL_ID)
      continue;
    if (lid>nb_var_value)
      throw FatalErrorException(A_FUNCINFO,"Bad variable");
    var_value[lid] = buffer[z];
  }
  info(4) << "End of read for variable '" << var->fullName()
          << "' path=" << path;
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Hdf5VariableReaderHelperBase::
_checkValidVariable(IVariable* var)
{
  if (var->dataType()==DT_Real && (var->dimension()==1) && var->itemFamily())
    return;
  throw FatalErrorException(A_FUNCINFO,
                            String::format("Bad variable '{0}'. Variable must be an item variable,"
                                           " have datatype 'Real' and dimension '1'",
                                           var->fullName()));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Lecture de variables au format HDF5.
 */
class Hdf5VariableReaderHelper
: public Hdf5VariableReaderHelperBase
{
 public:
  /*!
   * \todo permettre autre chose que 'Real' comme type de variable.
   */
  class TimeVariableInfoBase
  {
   public:
    TimeVariableInfoBase(const VariableItemReal& var)
    : m_variable(var),
      m_begin_variable(VariableBuildInfo(var.itemGroup().mesh(),String("Hdf5TimeVariableBegin")+var.name(),
                                         IVariable::PNoDump),var.itemGroup().itemKind()),
      m_end_variable(VariableBuildInfo(var.itemGroup().mesh(),String("Hdf5TimeVariableEnd")+var.name(),
                                       IVariable::PNoDump),var.itemGroup().itemKind()),
      m_current_index(-1), m_mesh_timestamp(-1)
      {
      }
   public:
    VariableItemReal& variable(){ return m_variable; }
   private:
    VariableItemReal m_variable;
   public:
    UniqueArray<TimePathPair> m_time_path_values;
    VariableItemReal m_begin_variable;
    VariableItemReal m_end_variable;

    /*!
     * \brief Contient l'indice dans le tableau des temps du temps actuellement lu.
     *
     * Cet indice vaut (-1) si aucun temps n'a été lu.
     */
    Integer m_current_index;
    //! Temps du maillage auquel on a lu cet index.
    Int64 m_mesh_timestamp;
  };


 public:

  Hdf5VariableReaderHelper(IMesh* mesh,const String& xml_file_name);
  ~Hdf5VariableReaderHelper();

 public:

  /*!
   * \brief Ouvre le fichier contenant les informations de lecture.
   *
   * \a is_start est vrai lors du démarrage d'un cas. Si ce n'est pas le cas,
   * il n'y a pas besoin de lire les variables d'initialisation.
   */
  void open(bool is_start);

  //! Lit les informations
  void readInit();

  //! Lecture et mise à jour des variables
  void readAndUpdateTimeVariables(Real wanted_time);

  //! Notification d'un retout-arrière 
  void notifyRestore();


 private:
	
  String m_xml_file_name;

  ScopedPtrT<IXmlDocumentHolder> m_xml_document_holder;
  UniqueArray<Hdf5VariableInfoBase*> m_init_variables;
  UniqueArray<TimeVariableInfoBase*> m_time_variables;

 private:

  //void _checkValidVariable(IVariable* var);
  //void _readVariable(IVariable* var,RealUniqueArray buffer,HFile& hfile,const String& path);
  void _readAndUpdateVariable(TimeVariableInfoBase* vi,Real wanted_time,HFile& hfile);

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Hdf5VariableReaderHelper::
Hdf5VariableReaderHelper(IMesh* mesh,const String& xml_file_name)
: Hdf5VariableReaderHelperBase(mesh)
, m_xml_file_name(xml_file_name)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Hdf5VariableReaderHelper::
~Hdf5VariableReaderHelper()
{
  for( Integer i=0, n=m_init_variables.size(); i<n; ++i )
    delete m_init_variables[i];
  m_init_variables.clear();
  for( Integer i=0, n=m_time_variables.size(); i<n; ++i )
    delete m_time_variables[i];
  m_time_variables.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Hdf5VariableReaderHelper::
open(bool is_start)
{
  if (m_xml_file_name.null())
    ARCANE_FATAL("No xml file specified");
  IIOMng* io_mng = m_mesh->parallelMng()->ioMng();
  m_xml_document_holder = io_mng->parseXmlFile(m_xml_file_name);
  if (!m_xml_document_holder.get())
    ARCANE_FATAL("Can not read file '{0}'",m_xml_file_name);

  XmlNode root_element = m_xml_document_holder->documentNode().documentElement();
  m_hdf5_file_name = root_element.attrValue("file-name",true);

  // Lecture des variables pour l'initialisation
  if (is_start){
    XmlNodeList variables_elem = root_element.children("init-variable");
    for( XmlNode elem : variables_elem ){
      String var_name = elem.attrValue("name",true);
      String var_family = elem.attrValue("family",true);
      String var_path = elem.attrValue("path",true);
      info() << "INIT_VARIABLE: name=" << var_name << " path=" << var_path
             << " family=" << var_family;
      Hdf5VariableInfoBase* var_info = Hdf5VariableInfoBase::create(m_mesh,var_name,var_family);
      var_info->setPath(var_path);
      m_init_variables.add(var_info);
    }
  }


  XmlNodeList variables_elem = root_element.children("time-variable");
  for( XmlNode elem : variables_elem ){
    String var_name = elem.attrValue("name",true);
    String var_family = elem.attrValue("family",true);
    info() << "TIME_VARIABLE: name=" << var_name << " family=" << var_family;
    IItemFamily* family = m_mesh->findItemFamily(var_family,true);
    IVariable* var = family->findVariable(var_name,false);
    if (!var){
      warning() << "TEMPORARY: Create variable from hdf5 file";
      VariableCellReal* vcr = new VariableCellReal(VariableBuildInfo(m_mesh,var_name));
      var = vcr->variable();
    }
    _checkValidVariable(var);
    VariableItemReal vir(VariableBuildInfo(m_mesh,var->name(),var->itemFamily()->name()),var->itemKind());
    TimeVariableInfoBase* var_info = new TimeVariableInfoBase(vir);

    XmlNodeList times_elem = elem.children("time-value");
    Real last_var_time = -1.0;
    for( XmlNode time_elem : times_elem ){
      String var_path = time_elem.attrValue("path",true);
      XmlNode var_time_node = time_elem.attr("global-time",true);
      Real var_time = var_time_node.valueAsReal(true);
      if (var_time<=last_var_time){
        fatal() << "Bad value for " << var_time_node.xpathFullName()
                << " current=" << var_time << " previous=" << last_var_time
                << ". current value should be greater than previous time.";
      }
      last_var_time = var_time;
      var_info->m_time_path_values.add(TimePathPair(var_time,var_path));
    }

    m_time_variables.add(var_info);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Hdf5VariableReaderHelper::
readInit()
{
  //TODO lancer exception en cas d'erreur.
  HFile hfile;

  for( Integer iz=0, izs=m_init_variables.size(); iz<izs; ++iz ){
    Hdf5VariableInfoBase* vi = m_init_variables[iz];
    IVariable* var = vi->variable();
    info() << "Hdf5VariableReader: init for variable name=" << var->fullName();
    vi->readVariable(hfile,m_hdf5_file_name,m_types,String(),var->data());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Hdf5VariableReaderHelper::
_readAndUpdateVariable(TimeVariableInfoBase* vi,Real wanted_time,HFile& hfile)
{
  VariableItemReal& var = vi->variable();

  ConstArrayView<TimePathPair> time_path_values(vi->m_time_path_values.view());
  Integer current_index = -1;
  Real begin_time = 0.0;
  Real end_time = 0.0;
  Integer nb_value = time_path_values.size();
  for( Integer i=0; i<nb_value; ++i ){
    begin_time = time_path_values[i].timeValue();
    if (wanted_time<begin_time){
      break;
    }
    current_index = i;
    if ((i+1)==nb_value){
      break;
    }
    if (math::isEqual(begin_time,wanted_time)){
      break;
    }
    end_time = time_path_values[i+1].timeValue();
    if (wanted_time>begin_time && wanted_time<end_time){
      break;
    }
  }
  // Ne fait rien si on n'est pas dans la table
  if (current_index<0)
    return;
  info(4) << " FIND TIME: var=" << var.variable()->fullName() << " current=" << wanted_time
          << " begin=" << begin_time << " end=" << end_time
          << " index=" << current_index;
  Int64 mesh_timestamp = var.variable()->meshHandle().mesh()->timestamp();
  bool need_read = current_index!=vi->m_current_index || vi->m_mesh_timestamp!=mesh_timestamp;
  if (nb_value==1 || (current_index+1)==nb_value){
    // On est à la fin de la table.
    // Dans ce cas, prend la valeur correspondant au dernier index sans faire
    // d'interpolation.
    if (need_read){
      RealUniqueArray buffer;
      String begin_path = vi->m_time_path_values[current_index].path();
      info() << "Hdf5VariableReaderHelper:: PATH=" << begin_path;
      _readVariable(vi->m_begin_variable.variable(),buffer,hfile,begin_path);
      vi->m_current_index = current_index;
      vi->m_mesh_timestamp = mesh_timestamp;
    }

    begin_time = vi->m_time_path_values[current_index].timeValue();
    VariableItemReal& begin_variable = vi->m_begin_variable;

    ENUMERATE_ITEM(iitem,var.itemGroup()){
      Real begin_value = begin_variable[iitem];
      var[iitem] = begin_value;
      if (m_is_verbose)
        info() << "Value for cell=" << (*iitem).uniqueId() << " var_value=" << var[iitem];
    }
  }
  else{
    if (need_read){
      RealUniqueArray buffer;
      String begin_path = vi->m_time_path_values[current_index].path();
      String end_path = vi->m_time_path_values[current_index+1].path();
      info(4) << "Hdf5VariableReaderHelper:: BEGIN_PATH=" << begin_path << " END_PATH=" << end_path;
      _readVariable(vi->m_begin_variable.variable(),buffer,hfile,begin_path);
      _readVariable(vi->m_end_variable.variable(),buffer,hfile,end_path);
      vi->m_current_index = current_index;
      vi->m_mesh_timestamp = mesh_timestamp;
    }

    begin_time = vi->m_time_path_values[current_index].timeValue();
    end_time = vi->m_time_path_values[current_index+1].timeValue();
    if (math::isEqual(begin_time,end_time))
      fatal() << "Hdf5VariableReaderHelper::_readAndUpdateVariable() "
              << " same value for begin and end time (value=" << begin_time << ")";
    Real ratio = (wanted_time - begin_time) / (end_time - begin_time);
    info(4) << " RATIO = " << ratio;
    VariableItemReal& begin_variable = vi->m_begin_variable;
    VariableItemReal& end_variable = vi->m_end_variable;

    ENUMERATE_ITEM(iitem,var.itemGroup()){
      Real begin_value = begin_variable[iitem];
      Real end_value = end_variable[iitem];
      var[iitem] = (end_value-begin_value)*ratio + begin_value;
      if (m_is_verbose){
        info() << "Value for cell=" << (*iitem).uniqueId()
               << " begin=" << begin_value << " end_value=" << end_value
               << " var_value=" << var[iitem];
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Hdf5VariableReaderHelper::
readAndUpdateTimeVariables(Real wanted_time)
{
  HFile hfile;

  for( Integer iz=0, izs=m_time_variables.size(); iz<izs; ++iz ){
    TimeVariableInfoBase* vi = m_time_variables[iz];
    _readAndUpdateVariable(vi,wanted_time,hfile);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Hdf5VariableReaderHelper::
notifyRestore()
{
  // Pour les variables qui dépendent du temps, indique que le temps
  // courant est invalide et qu'il faut le recharger
  for( Integer iz=0, izs=m_time_variables.size(); iz<izs; ++iz ){
    TimeVariableInfoBase* vi = m_time_variables[iz];
    vi->m_current_index = -1;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Lecture de variables au format HDF5.
 */
class Hdf5VariableReaderHelper2
: public Hdf5VariableReaderHelperBase
{
 public:
  /*!
   * Nouveau format permettant de relire n'importe quelle variable
   * (mais pour l'instant uniquement avec le type 'Real'...)
   * //TODO: traiter changement de maillage.
   */
  class TimeVariableInfoBase
  {
   public:
    TimeVariableInfoBase(Hdf5VariableInfoBase* var)
    : m_hdf5_var_info(var), m_current_index(-1), m_mesh_timestamp(-1)
    {
    }
    ~TimeVariableInfoBase()
    {
    }

   public:
    IVariable* variable(){ return  m_hdf5_var_info->variable(); }
    Hdf5VariableInfoBase* hdf5Info(){ return m_hdf5_var_info; }
    Real2 timeInterval() const
    {
      Integer n = m_time_path_values.size();
      if (n==0)
        return Real2(0.0,0.0);
      Real x = m_time_path_values[0].timeValue();
      Real y = m_time_path_values[n-1].timeValue();
      return Real2(x,y);
    }
    void rebuildData()
    {
      IVariable* v = variable();
      m_begin_data = v->data()->cloneRef();
      m_end_data = v->data()->cloneRef();
    }
   private:
    Hdf5VariableInfoBase* m_hdf5_var_info;
   public:
    UniqueArray<TimePathPair> m_time_path_values;
    Ref<IData> m_begin_data;
    Ref<IData> m_end_data;

    /*!
     * \brief Contient l'indice dans le tableau des temps du temps actuellement lu.
     *
     * Cet indice vaut (-1) si aucun temps n'a été lu.
     */
    Integer m_current_index;
    //! Temps du maillage auquel on a lu cet index.
    Int64 m_mesh_timestamp;
  };

  /*!
   * \brief Infos de correspondance entre les uids sauvés et ceux
   * du maillage courant pour le groupe \a group.
   */
  class CorrespondanceInfo : public Hdf5VariableInfoBase::ICorrespondanceFunctor
  {
   public:
    CorrespondanceInfo(const ItemGroup& group)
    : m_group(group),
      m_corresponding_uids(VariableBuildInfo(group.mesh(),String("CorrespondingUids_")+group.fullName(),IVariable::PNoRestore)),
      m_corresponding_hash(512,true)
    {
    }
   public:
    virtual Int64 getOldUniqueId(Int64 uid,Integer index)
    {
      ARCANE_UNUSED(index);
      HashTableMapT<Int64,Int64>::Data* uid_data = m_corresponding_hash.lookup(uid);
      if (!uid_data)
        throw FatalErrorException(A_FUNCINFO,
                                  String::format("Can not find corresponding uid item='{0}' group={1}",
                                                  uid,m_group.fullName()));
      Int64 old_uid = uid_data->value();
      return old_uid;
    }
   public:
    void updateHashMap()
    {
      m_corresponding_hash.clear();
      Integer nb_pair = m_corresponding_uids.size() / 2;
      for( Integer z=0; z<nb_pair; ++z )
        m_corresponding_hash.add(m_corresponding_uids[z*2],m_corresponding_uids[(z*2)+1]);
    }
   public:
    ItemGroup m_group;
    VariableArrayInt64 m_corresponding_uids;
    HashTableMapT<Int64,Int64> m_corresponding_hash;
  };

 public:

  Hdf5VariableReaderHelper2(IMesh* mesh,const String& hdf5_file_name);
  ~Hdf5VariableReaderHelper2();

 public:

  /*!
   * \brief Spécifie les variables qu'on souhaite relire.
   *
   * Cette méthode doit être appelée avant open(). Si cette méthode n'est
   * pas appelée, on essaie de relire toutes les variables sauvegardées dans
   * le fichier.
   */
  void setVariables(ConstArrayView<IVariable*> vars)
  {
    m_wanted_vars = vars;
  }

  /*!
   * \brief Ouvre le fichier contenant les informations de lecture.
   *
   * \a is_start est vrai lors du démarrage d'un cas. Si ce n'est pas le cas,
   * il n'y a pas besoin de lire les variables d'initialisation.
   */
  void open(bool is_start);

  //! Lecture et mise à jour des variables
  void readAndUpdateTimeVariables(Real wanted_time);

  //! Notification d'un retout-arrière 
  void notifyRestore();

  Real2 timeInterval(IVariable* var)
  {
    for( Integer i=0, n=m_time_variables.size(); i<n; ++i ){
      TimeVariableInfoBase* vinfo = m_time_variables[i];
      if (vinfo->variable()==var)
        return vinfo->timeInterval();
    }
    return Real2(0.0,0.0);
  }

 protected:

 private:
	
  UniqueArray<IVariable*> m_wanted_vars;
  ScopedPtrT<IXmlDocumentHolder> m_xml_document_holder;
  UniqueArray<TimeVariableInfoBase*> m_time_variables;
  std::map<String,CorrespondanceInfo*> m_correspondance_map;

 private:

  template <typename DataType>
  void _readAndUpdateVariable(TimeVariableInfoBase* vi,Real wanted_time,HFile& hfile);
  bool _isWanted(const String& var_name,const String& var_family);
  void _checkCreateCorrespondance(Hdf5VariableInfoBase* var,HFile& file_id,const String& group_path,bool is_start);
  void _createCorrespondance(IVariable* var,CorrespondanceInfo* ci,Int64ConstArrayView saved_uids,Real3ConstArrayView saved_centers);

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Hdf5VariableReaderHelper2::
Hdf5VariableReaderHelper2(IMesh* mesh,const String& hdf5_file_name)
: Hdf5VariableReaderHelperBase(mesh)
{
  m_hdf5_file_name = hdf5_file_name;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Hdf5VariableReaderHelper2::
~Hdf5VariableReaderHelper2()
{
  for( Integer i=0, n=m_time_variables.size(); i<n; ++i )
    delete m_time_variables[i];
  m_time_variables.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Hdf5VariableReaderHelper2::
open(bool is_start)
{
  if (m_hdf5_file_name.null())
    throw FatalErrorException(A_FUNCINFO,"No hdf5 file specified");

  HFile file_id;
  IParallelMng* pm = m_mesh->parallelMng();
  bool is_master = pm->isMasterIO();
  ByteUniqueArray xml_bytes;
  if (is_master){
    file_id.openRead(m_hdf5_file_name);
    Hdf5Utils::StandardArrayT<Byte> v(file_id.id(),"Infos");
    v.directRead(m_types,xml_bytes);
  }
  info(5) << "XML_DATA len=" << xml_bytes.size() << " data=" << xml_bytes << "__EOF";
  pm->broadcastMemoryBuffer(xml_bytes,pm->masterIORank());
  
  IIOMng* io_mng = m_mesh->parallelMng()->ioMng();
  m_xml_document_holder = io_mng->parseXmlBuffer(xml_bytes,m_hdf5_file_name);
  if (!m_xml_document_holder.get())
    ARCANE_FATAL("Can not XML data from file '{0}'",m_hdf5_file_name);

  XmlNode root_element = m_xml_document_holder->documentNode().documentElement();
  //m_hdf5_file_name = root_element.attrValue("file-name",true);

  XmlNodeList variables_elem = root_element.children("time-variable");
  for( const auto& elem : variables_elem ){
    String var_name = elem.attrValue("name",true);
    String var_family = elem.attrValue("family",true);
    info(4) << "TIME_VARIABLE: name=" << var_name << " family=" << var_family;
    if (!_isWanted(var_name,var_family))
      continue;
    //TODO: creer la variable si elle n'existe pas ou faire quelque chose (exception...)
    Hdf5VariableInfoBase* var_info = Hdf5VariableInfoBase::create(m_mesh,var_name,var_family);
    //TODO: tmp
    String group_name = var_info->variable()->itemGroupName();
    String index_path = String("Index") + 1;
    String group_path = index_path + "/Groups/" + group_name;
    _checkCreateCorrespondance(var_info,file_id,group_path,is_start);
    
    TimeVariableInfoBase* time_var_info = new TimeVariableInfoBase(var_info);
    
    XmlNodeList times_elem = elem.children("time-value");
    Real last_var_time = -1.0;
    for( const auto& time_elem : times_elem ){
      String var_path = time_elem.attrValue("path",true);
      XmlNode var_time_node = time_elem.attr("global-time",true);
      Real var_time = var_time_node.valueAsReal(true);
      if (var_time<=last_var_time){
        fatal() << "Bad value for " << var_time_node.xpathFullName()
                << " current=" << var_time << " previous=" << last_var_time
                << ". current value should be greater than previous time.";
      }
      last_var_time = var_time;
      time_var_info->m_time_path_values.add(TimePathPair(var_time,var_path));
    }

    m_time_variables.add(time_var_info);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Hdf5VariableReaderHelper2::
_checkCreateCorrespondance(Hdf5VariableInfoBase* var_info,HFile& file_id,const String& group_path,bool is_start)
{
  IVariable* var = var_info->variable();
  // Vérifie si la correspondance existe déjà.
  ItemGroup group = var->itemGroup();
  CorrespondanceInfo* ci = 0;
  std::map<String,CorrespondanceInfo*>::const_iterator iter = m_correspondance_map.find(group.fullName());
  if (iter==m_correspondance_map.end()){
    ci = new CorrespondanceInfo(group);
    if (is_start){
      Int64UniqueArray saved_uids;
      Real3UniqueArray saved_centers;
      var_info->readGroupInfo(file_id,m_types,group_path,saved_uids,saved_centers);
      _createCorrespondance(var,ci,saved_uids,saved_centers);
    }
    ci->updateHashMap();
    m_correspondance_map.insert(std::make_pair(group.fullName(),ci));
  }
  else
    ci = iter->second;
  var_info->setCorrespondanceFunctor(ci);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Recherche à quel entité du maillage sauvegardé correspondant
 * une entité du maillage actuel.
 *
 * Lorsqu'on relit les valeurs des variables d'un fichier, on supporte le fait
 * que le maillage actuel n'est pas forcément le même que le maillage
 * sauvegardé. Dans ce cas, les uniqueId des entités ne correspondent pas.
 * Il faut donc chercher à quel entité sauvée correspondant chaque entité
 * du maillage actuel. Pour savoir laquelle utiliser, on prend celle
 * du maillage d'origine la plus proche de celle du maillage actuel.
 * Comme seules les coordonnées initiales sont sauvées, cela ne fonctionne
 * correctement que lors de l'initialisation. Il ne faut pas faire
 * ce traitement en reprise. On sauvegarde donc cette information dans
 * une variable.
 * TODO: Il faudrait pouvoir utiliser un autre algorithme que juste
 * l'entité la plus proche.
 */
void Hdf5VariableReaderHelper2::
_createCorrespondance(IVariable* var,CorrespondanceInfo* ci,Int64ConstArrayView saved_uids,
                      Real3ConstArrayView saved_centers)
{
  IMesh* mesh = var->meshHandle().mesh();
  ItemGroup group = var->itemGroup();
  IParallelMng* pm = mesh->parallelMng();
 
  Int64UniqueArray corresponding_uids;
  Integer nb_orig_item = saved_uids.size();
  VariableNodeReal3& nodes_coords(mesh->nodesCoordinates());
  ENUMERATE_ITEM(iitem,group){
    Real3 item_center;
    ItemUniqueId item_uid = (*iitem).uniqueId();
    if ((*iitem).isItemWithNodes()){
      ItemWithNodes item = (*iitem).toItemWithNodes();
      if (!item.isOwn())
        continue;
      Integer nb_node = item.nbNode();
      for( NodeEnumerator inode(item.nodes()); inode.hasNext(); ++inode ){
        item_center += nodes_coords[inode];
      }
      item_center /= nb_node;
    }
    else{
      Node node = (*iitem).toNode();
      item_center = nodes_coords[node];
    }

    // Recherche l'entité la plus proche.
    Real min_dist = FloatInfo<Real>::maxValue();
    Integer min_index = -1;
    for( Integer z=0; z<nb_orig_item; ++z ){
      Real d = (item_center - saved_centers[z]).squareNormL2();
      if (d<min_dist){
        min_dist = d;
        min_index = z;
      }
    }
    if (min_index==(-1))
      throw FatalErrorException(A_FUNCINFO,"Can not find old unique id");
    info() << "FIND NEAREST my_uid=" << item_uid << " orig_uid=" << saved_uids[min_index]
           << " d^2=" << min_dist;
    corresponding_uids.add(item_uid);
    corresponding_uids.add(saved_uids[min_index]);
  }

  // Pour l'instant et pour se simplifier la vie en cas de repartitionnement
  // de maillage, on récupère toutes les infos des autres sous-domaine.
  // Ce n'est pas idéal car cela duplique les infos chez tout le monde,
  // mais cela permet de ne pas avoir à gérer le repartitionnement.
  // A terme, il faudrait répartir ces infos sur chaque proc et les
  // regrouper au moment de mettre à jour la table de hashage lors
  // d'un changement de maillage.
  Int64UniqueArray global_uids;
  pm->allGatherVariable(corresponding_uids,global_uids);
  ci->m_corresponding_uids.resize(global_uids.size());
  ci->m_corresponding_uids.copy(global_uids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool Hdf5VariableReaderHelper2::
_isWanted(const String& var_name,const String& var_family)
{
  Integer nb_var = m_wanted_vars.size();
  if (nb_var==0)
    return true;
  for( Integer i=0, n=m_wanted_vars.size(); i<n; ++i ){
    IVariable* v = m_wanted_vars[i];
    if (v->name()==var_name && v->itemFamilyName()==var_family)
      return true;
  }
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType>
void Hdf5VariableReaderHelper2::
_readAndUpdateVariable(TimeVariableInfoBase* vi,Real wanted_time,HFile& hfile)
{
  //VariableItemReal& var = vi->variable();

  ConstArrayView<TimePathPair> time_path_values(vi->m_time_path_values.view());
  Integer current_index = -1;
  Real begin_time = 0.0;
  Real end_time = 0.0;
  Integer nb_value = time_path_values.size();
  for( Integer i=0; i<nb_value; ++i ){
    begin_time = time_path_values[i].timeValue();
    if (wanted_time<begin_time){
      break;
    }
    current_index = i;
    if ((i+1)==nb_value){
      // Atteint la fin de la table. Ne fait rien.
      current_index = -1;
      break;
    }
    if (math::isEqual(begin_time,wanted_time)){
      break;
    }
    end_time = time_path_values[i+1].timeValue();
    if (wanted_time>begin_time && wanted_time<end_time){
      break;
    }
  }
  // Ne fait rien si on n'est pas dans la table
  if (current_index<0)
    return;

  //TODO utiliser le modifiedTime du groupe de la variable plutot que du maillage
  IVariable* variable = vi->variable();
  Int64 mesh_timestamp = variable->meshHandle().mesh()->timestamp();

  // Il faut relire les infos si on change d'indice dans la table
  // ou si le maillage change
  bool need_read = current_index!=vi->m_current_index || vi->m_mesh_timestamp!=mesh_timestamp;
  if (need_read)
    vi->rebuildData();

  String ids_hpath = String("Index1/Groups/") + variable->itemGroupName() + "_Ids";
  info(4) << " FIND TIME: current=" << wanted_time << " begin=" << begin_time << " end=" << end_time
          << " index=" << current_index << "ids_path=" << ids_hpath << " var=" << variable->fullName();
  Hdf5VariableInfoBase* var_info = vi->hdf5Info();
  IArrayDataT<DataType>* true_begin_data = dynamic_cast<IArrayDataT<DataType>*>(vi->m_begin_data.get());
  IArrayDataT<DataType>* true_end_data = dynamic_cast<IArrayDataT<DataType>*>(vi->m_end_data.get());
  IArrayDataT<DataType>* true_data = dynamic_cast<IArrayDataT<DataType>*>(variable->data());
  info(4) << "DATA: begin=" << vi->m_begin_data.get() << " end=" << vi->m_end_data.get() << " current=" << variable->data();
  info(4) << "TRUEDATA: begin=" << true_begin_data << " end=" << true_end_data << " current=" << true_data;
  if (!true_data || !true_end_data || !true_begin_data){
    throw FatalErrorException(A_FUNCINFO,"variable data can not be cast to type IArrayDataT");
  }
  bool is_partial = variable->isPartial();

  if (need_read){
    String begin_path = vi->m_time_path_values[current_index].path();
    String end_path = vi->m_time_path_values[current_index+1].path();
    info(4) << "Hdf5VariableReaderHelper2:: Reading new index BEGIN_PATH=" << begin_path << " END_PATH=" << end_path;
    var_info->setPath(begin_path);
    var_info->readVariable(hfile,m_hdf5_file_name,m_types,ids_hpath,vi->m_begin_data.get());
    var_info->setPath(end_path);
    var_info->readVariable(hfile,m_hdf5_file_name,m_types,ids_hpath,vi->m_end_data.get());
  }

  vi->m_current_index = current_index;
  vi->m_mesh_timestamp = mesh_timestamp;

  begin_time = vi->m_time_path_values[current_index].timeValue();
  end_time = vi->m_time_path_values[current_index+1].timeValue();
  if (math::isEqual(begin_time,end_time))
    throw FatalErrorException(A_FUNCINFO,
                              String::format("same value for begin and end time (value={0})",
                                             begin_time));
  Real ratio = (wanted_time - begin_time) / (end_time - begin_time);
  info(4) << " BeginTime=" << begin_time << " wanted=" << wanted_time << " end_time=" << end_time
          << " ratio = " << ratio;
  {
    ConstArrayView<DataType> begin_values_view = true_begin_data->view();
    ConstArrayView<DataType> end_values_view = true_end_data->view();
    ArrayView<DataType> values_view = true_data->view();
    ENUMERATE_ITEM(iitem,variable->itemGroup()){
      Int32 lid = (is_partial) ? iitem.index() : iitem.itemLocalId();
      DataType begin_value = begin_values_view[lid];
      DataType end_value = end_values_view[lid];
      values_view[lid] = (end_value-begin_value)*ratio + begin_value;
      if (m_is_verbose){
        info() << "Value for cell=" << (*iitem).uniqueId()
               << " begin=" << begin_value << " end_value=" << end_value
               << " var_value=" << values_view[lid];
      }
    }
  }
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Hdf5VariableReaderHelper2::
readAndUpdateTimeVariables(Real wanted_time)
{
  HFile hfile;

  for( Integer iz=0, izs=m_time_variables.size(); iz<izs; ++iz ){
    TimeVariableInfoBase* vi = m_time_variables[iz];
    IVariable* variable=vi->variable();
    switch (variable->dataType()) {
    case DT_Real:
      _readAndUpdateVariable<Real>(vi,wanted_time,hfile);
      break;
    case DT_Real3:
      _readAndUpdateVariable<Real3>(vi,wanted_time,hfile);
      break;
    default:
      throw NotSupportedException(A_FUNCINFO,"Bad variable datatype (only Real and Real3 are supported)");
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Hdf5VariableReaderHelper2::
notifyRestore()
{
  // Pour les variables qui dépendent du temps, indique que le temps
  // courant est invalide et qu'il faut le recharger
  for( Integer iz=0, izs=m_time_variables.size(); iz<izs; ++iz ){
    TimeVariableInfoBase* vi = m_time_variables[iz];
    vi->m_current_index = -1;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Lecture de variables au format HDF5.
 */
class Hdf5VariableReader
: public ArcaneHdf5VariableReaderObject
{
 public:

  explicit Hdf5VariableReader(const ServiceBuildInfo& sbi);
  ~Hdf5VariableReader() override;

 public:

  void build() override
  {
  }

  void onTimeLoopStartInit() override
  {
    _load(true);
    for( Integer i=0, is=m_readers.size(); i<is; ++i ){
      m_readers[i]->readInit();
    }
  }
  void onTimeLoopContinueInit() override
  {
    // En reprise, il faut charger les variables mais pas les relire
    _load(false);
  }
  void onTimeLoopEndLoop() override {}
  void onTimeLoopRestore() override
  {
    for( Integer i=0, is=m_readers.size(); i<is; ++i ){
      m_readers[i]->notifyRestore();
    }
  }
  void onTimeLoopBeginLoop() override
  {
    Real current_time = subDomain()->commonVariables().globalTime();
    for( Integer i=0, is=m_readers.size(); i<is; ++i ){
      m_readers[i]->readAndUpdateTimeVariables(current_time);
    }
  }

 private:

  void _load(bool is_start)
  {
    IMeshMng* mm = subDomain()->meshMng();
    for( Integer i=0, is=options()->read.size(); i<is; ++i ){
      String file_name = options()->read[i]->fileName();
      String mesh_name = options()->read[i]->meshName();
      info() << "Hdf5VariableReader: FILE_INFO: mesh=" << mesh_name << " file_name=" << file_name;
      {
        IMesh* mesh = mm->findMeshHandle(mesh_name).mesh();
        Hdf5VariableReaderHelper* sd = new Hdf5VariableReaderHelper(mesh,file_name);
        m_readers.add(sd);
      }
    }
    info() << "Hdf5VariableReader: Nb reader =" << m_readers.size();
    for( Integer i=0, is=m_readers.size(); i<is; ++i ){
      m_readers[i]->open(is_start);
    }
  }

 private:
  UniqueArray<Hdf5VariableReaderHelper*> m_readers;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Hdf5VariableReader::
Hdf5VariableReader(const ServiceBuildInfo& sbi)
: ArcaneHdf5VariableReaderObject(sbi)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Hdf5VariableReader::
~Hdf5VariableReader()
{
  for( Integer i=0, is=m_readers.size(); i<is; ++i )
    delete m_readers[i];
  m_readers.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Lecture de variables au format HDF5.
 */
class ManualHdf5VariableReader
: public BasicService
, public IVariableReader
{
 public:

  ManualHdf5VariableReader(const ServiceBuildInfo& sbi);
  ~ManualHdf5VariableReader();

 public:

  virtual void build()
  {
  }

  virtual void read(IVariable* var)
  {
    ARCANE_UNUSED(var);
  }

  virtual void initialize(bool is_start)
  {
    pwarning() << "Reading variable from HDF5 file does not work with mesh load balancing";
    if (m_base_file_name.null())
      m_base_file_name = "data";
    String file_name = m_base_file_name + ".h5";
    Directory dir(m_directory_name);
    String full_file_name = dir.file(file_name);
    m_helper = new Hdf5VariableReaderHelper2(mesh(),full_file_name);
    m_helper->setVariables(m_variables);
    m_helper->open(is_start);
  }

  virtual void updateVariables(Real wanted_time)
  {
    m_helper->readAndUpdateTimeVariables(wanted_time);
  }

  virtual void setBaseDirectoryName(const String& path)
  {
    m_directory_name = path;
  }

  virtual void setBaseFileName(const String& path)
  {
    m_base_file_name = path;
  }

  virtual void setVariables(VariableCollection vars)
  {
    for( VariableCollection::Enumerator ivar(vars); ++ivar; )
      m_variables.add(*ivar);
  }
  virtual Real2 timeInterval(IVariable* var)
  {
    arcaneCheckNull(m_helper);
    arcaneCheckNull(var);
    return m_helper->timeInterval(var);
  }

 public:
  Hdf5VariableReaderHelper2* m_helper;
  String m_directory_name;
  String m_base_file_name;
  UniqueArray<IVariable*> m_variables;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ManualHdf5VariableReader::
ManualHdf5VariableReader(const ServiceBuildInfo& sbi)
: BasicService(sbi)
, m_helper(0)
, m_directory_name(".")
, m_base_file_name("data")
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ManualHdf5VariableReader::
~ManualHdf5VariableReader()
{
  delete m_helper;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Lecture de variables au format HDF5 via un descripteur XML
 */
class OldManualHdf5VariableReader
: public BasicService
, public IVariableReader
{
 public:

  OldManualHdf5VariableReader(const ServiceBuildInfo& sbi)
  : BasicService(sbi), m_reader(nullptr){}
  ~OldManualHdf5VariableReader()
  {
    delete m_reader;
  }

 public:

  virtual void build()
  {
  }

 public:
  
  virtual void read(IVariable* var)
  {
    ARCANE_UNUSED(var);
    throw NotSupportedException(A_FUNCINFO);
  }

  virtual void initialize(bool is_start)
  {
    _load(is_start);
  }

  virtual void updateVariables(Real wanted_time)
  {
    m_reader->readAndUpdateTimeVariables(wanted_time);
  }

  virtual void setBaseDirectoryName(const String& path)
  {
    m_directory_name = path;
  }

  virtual void setBaseFileName(const String& path)
  {
    m_base_file_name = path;
  }

  virtual void setVariables(VariableCollection vars)
  {
    ARCANE_UNUSED(vars);
    throw NotSupportedException(A_FUNCINFO);
  }

  virtual Real2 timeInterval(IVariable* var)
  {
    ARCANE_UNUSED(var);
    throw NotSupportedException(A_FUNCINFO);
  }

 private:

  void _load(bool is_start)
  {
    ARCANE_UNUSED(is_start);

    Directory dir(m_directory_name);
    String file_name = dir.file(m_base_file_name);

    info() << "OldManualHdf5VariableReader: FILE_INFO: file_name=" << file_name;
    IMesh* mesh = subDomain()->defaultMesh();
    Hdf5VariableReaderHelper* sd = new Hdf5VariableReaderHelper(mesh,file_name);
    m_reader = sd;
  }

 private:
  Hdf5VariableReaderHelper* m_reader;
  String m_directory_name;
  String m_base_file_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


ARCANE_REGISTER_SERVICE_HDF5VARIABLEREADER(Hdf5VariableReader,
                                           Hdf5VariableReader);

ARCANE_REGISTER_SERVICE(ManualHdf5VariableReader,
                        ServiceProperty("Hdf5VariableReader",ST_SubDomain),
                        ARCANE_SERVICE_INTERFACE(IVariableReader));

ARCANE_REGISTER_SERVICE(OldManualHdf5VariableReader,
                        ServiceProperty("OldManualHdf5VariableReader",ST_SubDomain),
                        ARCANE_SERVICE_INTERFACE(IVariableReader));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
