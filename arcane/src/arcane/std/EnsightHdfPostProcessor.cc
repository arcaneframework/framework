// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* EnsightHdfPostProcessor.cc                                  (C) 2000-2023 */
/*                                                                           */
/* Exportations des fichiers au format Ensight HDF.                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/Collection.h"
#include "arcane/utils/Enumerator.h"
#include "arcane/utils/Iostream.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/CheckedConvert.h"

#include "arcane/PostProcessorWriterBase.h"

#include "arcane/std/Hdf5Utils.h"
#include "arcane/std/EnsightHdfPostProcessor_axl.h"

#include "arcane/FactoryService.h"
#include "arcane/IDataWriter.h"
#include "arcane/IMesh.h"
#include "arcane/IMeshSubMeshTransition.h"
#include "arcane/IData.h"
#include "arcane/ISerializedData.h"
#include "arcane/IItemFamily.h"
#include "arcane/VariableCollection.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Hdf5Utils;

#if 0
void _addFunc2(Integer item_type,Integer nb_node,const String& ensight_name)
{
}

void _addFunc()
{
  // NOTE: il est important que les éléments de type 'nfaced'
  // et 'nsided' soient contigues car Ensight doit sauver
  // ensemble leur valeurs
  _addFunc2(IT_Line2,2,"bar2"); // Bar
  _addFunc2(IT_Triangle3,3,"tria3"); // Triangle
  _addFunc2(IT_Quad4,4,"quad4"); // Quandrangle
  _addFunc2(IT_Pentagon5,5,"nsided"); // Pentagone
  _addFunc2(IT_Hexagon6,6,"nsided"); // Hexagone
  _addFunc2(IT_Heptagon7,7,"nsided"); // Heptagone
  _addFunc2(IT_Octogon8,8,"nsided"); // Octogone
  _addFunc2(IT_Tetraedron4,4,"tetra4"); // Tetra
  _addFunc2(IT_Pyramid5,5,"pyramid5"); // Pyramide
  _addFunc2(IT_Pentaedron6,6,"penta6"); // Penta
  _addFunc2(IT_Hexaedron8,8,"hexa8"); // Hexa
  _addFunc2(IT_Heptaedron10,10,"nfaced"); // Wedge7
  _addFunc2(IT_Octaedron12,12,"nfaced"); // Wedge8
  _addFunc2(IT_Enneedron14,14,"nfaced"); // Wedge9
  _addFunc2(IT_Decaedron16,16,"nfaced"); // Wedge10
}
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class EnsightHdfDataWriter
: public TraceAccessor
, public IDataWriter
{
 public:

  EnsightHdfDataWriter(IMesh* mesh,ItemGroupCollection groups,
                       RealConstArrayView times);

 public:

  virtual void beginWrite(const VariableCollection& vars);
  virtual void endWrite();
  virtual void setMetaData(const String& meta_data);
  virtual void write(IVariable* var,IData* data);

 private:

  IMesh* m_mesh;
  
  //! Liste des groupes à sauver
  ItemGroupCollection m_groups;

  //! Liste des temps
  RealConstArrayView m_times;
  
  Integer m_time_index;

  //! Nom du fichier HDF courant
  String m_filename;
  
  //! Identifiant HDF du fichier 
  HFile m_file_id;

  HGroup m_geometry_top_group;

  HGroup m_variable_top_group;

  Integer m_variable_index;

 private:
  
  void _addRealAttribute(Hid& hid,const char* name,double value);
  void _addRealArrayAttribute(Hid& hid,const char* name,RealConstArrayView values);
  void _addIntegerAttribute(Hid& hid,const char* name,int value);
  void _addStringAttribute(Hid& hid,const char* name,const String& value,hsize_t len);
  void _saveGroup(const ItemGroup& group,HGroup& domain_group);
  void _saveVariableOnGroup(IVariable* var,IData* data,const ItemGroup& group,RealArrayView min_values,
                            RealArrayView max_values,HGroup& domain_group);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EnsightHdfDataWriter::
EnsightHdfDataWriter(IMesh* mesh,ItemGroupCollection groups,RealConstArrayView times)
: TraceAccessor(mesh->traceMng())
, m_mesh(mesh)
, m_groups(groups)
, m_times(times)
, m_time_index(0)
, m_variable_index(0)
{
  m_time_index = times.size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EnsightHdfDataWriter::
beginWrite(const VariableCollection& vars)
{
  warning() << "L'implémentation du format 'EnsightHdf' n'est pas encore opérationnelle";
  
  StringBuilder sb("hdf5_ensight_");
  sb += m_time_index;
  m_filename = sb.toString();

  m_variable_index = 0;

  info() << "ENSIGHT HDF BEGIN WRITE file=" << m_filename;

  H5open();

  m_file_id.openTruncate(m_filename);
  HGroup top_group;
  top_group.create(m_file_id,"model");

  _addStringAttribute(top_group,"hdfrwRelease","0.9.0",20);
  _addIntegerAttribute(top_group,"nParts",m_groups.count());
  _addIntegerAttribute(top_group,"nVariables",vars.count());
  _addIntegerAttribute(top_group,"nDomains",1);
  _addIntegerAttribute(top_group,"hasNodeIDs",0);
  _addIntegerAttribute(top_group,"hasElementIDs",0);
  _addIntegerAttribute(top_group,"timeStep",1);
  // La valeur 2 indique que la connectivité est décrite dans le fichier
  //_addIntegerAttribute(top_group,"geomChangingFlag",2); // CONN_HDFRW
  //_addIntegerAttribute(top_group,"writingConn",0);

  _addIntegerAttribute(top_group,"geomChangingFlag",1); // COORDS_ONLY_HDFRW
  _addIntegerAttribute(top_group,"writingConn",1);

  _addRealAttribute(top_group,"timeValue",m_times[m_time_index-1]);
  _addStringAttribute(top_group,"modelDescription1","MyModel1",80);
  _addStringAttribute(top_group,"modelDescription2","MyModel2",80);
  _addStringAttribute(top_group,"lastGeomFile","None",256);
  
  m_geometry_top_group.create(top_group,"Geometry");
  m_variable_top_group.create(top_group,"Variables");

  {
    Integer index = 1;
    for( ItemGroupCollection::Enumerator igroup(m_groups); ++igroup; ++index ){
      const ItemGroup& group = *igroup;
    
      HGroup geometry_group;
      String part_name(String("geom_part_")+index);
      geometry_group.create(m_geometry_top_group,part_name.localstr());

      HGroup domain_group;
      String domain_name("geom_domain_1");
      domain_group.create(geometry_group,domain_name.localstr());
      
      // La valeur 0 indique qu'il s'agit d'un maillage non-structuré
      _addIntegerAttribute(geometry_group,"structuredFlag",0); // UNSTRUCTURED_HDFRW
      _addStringAttribute(geometry_group,"partName",group.name().localstr(),80);
      _addStringAttribute(geometry_group,"partDescription","None",80);
      _saveGroup(group,domain_group);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EnsightHdfDataWriter::
_saveGroup(const ItemGroup& group,HGroup& domain_group)
{
  info() << "SAVE GROUP name=" << group.name();
  //coord_dataset.open(m_domain_group,"coordinates");

  VariableNodeReal3 nodes_coordinates(m_mesh->toPrimaryMesh()->nodesCoordinates());
  Real3ConstArrayView nodes_coordinates_array = nodes_coordinates.asArray();

  // Premièrement, détermine les noeuds utilisés pour ce groupe, et
  // leur assigne un index local (qui doit commencer à un pour Ensight)
  IItemFamily* node_family = m_mesh->nodeFamily();
  Integer max_index = node_family->maxLocalId();
  Int32UniqueArray node_local_indexes(max_index);
  node_local_indexes.fill(NULL_ITEM_LOCAL_ID);
  Real3UniqueArray node_local_coords;
  Integer nb_local_node = 0;

  {
    ENUMERATE_ITEM(iitem,group){
      const Item& _item = *iitem;
      const ItemWithNodes& item = _item.toItemWithNodes();
      //Integer index = 0;
      for( NodeLocalId inode_lid : item.nodeIds() ){
        node_local_indexes[inode_lid] = 0;
      }
    }
    Integer current_index = 1;
    for( Integer i=0; i<max_index; ++i ){
      if (node_local_indexes[i]!=NULL_ITEM_LOCAL_ID){
        node_local_indexes[i] = current_index;
        ++current_index;
        node_local_coords.add(nodes_coordinates_array[i]);
      }
    }
    nb_local_node = current_index - 1;
  }

  info() << "TODO: save extents for geometry";

  // Sauve les coordonnées des noeuds
  {
    HDataset coord_dataset;

    Real3* ptr = node_local_coords.data();

#if 0
    ENUMERATE_NODE(inode,all_nodes){
      const Node& node = *inode;
      info() << " NODE uid=" << node.uniqueId() << " lid=" << node.localId();
    }
#endif

    hsize_t dims[1];
    dims[0] = nb_local_node*3;
    
    HSpace hspace;
    hspace.createSimple(1,dims);

    coord_dataset.create(domain_group,"coordinates",H5T_NATIVE_FLOAT,hspace,H5P_DEFAULT);
    coord_dataset.write(H5T_NATIVE_DOUBLE,(Real*)ptr);

    _addIntegerAttribute(coord_dataset,"layout",1); // INTERLACE_ORDER_1D
    _addIntegerAttribute(coord_dataset,"nNodes",nb_local_node);
  }

  // Sauve les éléments
  // ATTENTION: suppose uniquement des hexa8
  {
    HDataset item_dataset;

    ItemGroup all_items = group;
    Integer nb_item = all_items.size();
    Int32UniqueArray item_node_ids;
    ENUMERATE_ITEM(iitem,all_items){
      const Item& _item = *iitem;
      const ItemWithNodes& item = _item.toItemWithNodes();
      for( NodeLocalId inode_lid : item.nodeIds() ){
        item_node_ids.add(node_local_indexes[inode_lid]);
      }
    }
    hsize_t dims[1];
    dims[0] = nb_item*8;
    
    HSpace hspace;
    hspace.createSimple(1,dims);

    item_dataset.create(domain_group,"hexa8",H5T_NATIVE_INT,hspace,H5P_DEFAULT);
    item_dataset.write(H5T_NATIVE_INT,item_node_ids.data());

    _addIntegerAttribute(item_dataset,"layout",1); // INTERLACE_ORDER_1D
    // NOTE: dans le pdf de hdf_rw, l'attribut s'appelle 'nElems' mais il
    // faut mettre 'numof'
    _addIntegerAttribute(item_dataset,"numof",nb_item);
  }

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EnsightHdfDataWriter::
_addIntegerAttribute(Hid& hid,const char* name,int value)
{
  hid_t aid  = H5Screate(H5S_SCALAR);
  hid_t attr = H5Acreate2(hid.id(), name, H5T_NATIVE_INT, aid, H5P_DEFAULT,H5P_DEFAULT);
  if (attr<0)
    throw FatalErrorException(A_FUNCINFO,String("Can not create attribute ")+name);
  int ret  = H5Awrite(attr, H5T_NATIVE_INT, &value);
  ret  = H5Sclose(aid);
  ret  = H5Aclose(attr);
  if (ret<0)
    throw FatalErrorException(A_FUNCINFO,String("Can not write attribute ")+name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EnsightHdfDataWriter::
_addRealAttribute(Hid& hid,const char* name,double value)
{
  hid_t aid  = H5Screate(H5S_SCALAR);
  hid_t attr = H5Acreate2(hid.id(),name, H5T_NATIVE_FLOAT, aid, H5P_DEFAULT,H5P_DEFAULT);
  if (attr<0)
    throw FatalErrorException(String("Can not create attribute ")+name);
  int ret  = H5Awrite(attr, H5T_NATIVE_DOUBLE, &value);
  ret  = H5Sclose(aid);
  ret  = H5Aclose(attr);
  if (ret<0)
    throw FatalErrorException(A_FUNCINFO,String("Can not write attribute ")+name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EnsightHdfDataWriter::
_addRealArrayAttribute(Hid& hid,const char* name,RealConstArrayView values)
{
  hsize_t len = values.size();
  hid_t aid  = H5Screate_simple(1, &len, 0);
  //hid_t aid  = H5Screate(H5S_SCALAR);
  hid_t attr = H5Acreate2(hid.id(),name, H5T_NATIVE_FLOAT, aid, H5P_DEFAULT,H5P_DEFAULT);
  if (attr<0)
    throw FatalErrorException(String("Can not create attribute ")+name);
  int ret  = H5Awrite(attr, H5T_NATIVE_DOUBLE, values.data());
  ret  = H5Sclose(aid);
  ret  = H5Aclose(attr);
  if (ret<0)
    throw FatalErrorException(A_FUNCINFO,String("Can not write attribute ")+name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EnsightHdfDataWriter::
_addStringAttribute(Hid& hid,const char* name,const String& value,hsize_t len)
{
  UniqueArray<char> buf(CheckedConvert::toInteger(len));
  buf.fill('\0');
  const char* value_str = value.localstr();
  strncpy(buf.data(),value_str,len-1);
  
  hid_t aid  = H5Screate_simple(1, &len, NULL);
  hid_t attr = H5Acreate2(hid.id(),name, H5T_NATIVE_CHAR, aid, H5P_DEFAULT,H5P_DEFAULT);
  if (attr<0)
    throw FatalErrorException(String("Can not create attribute ")+name);
  int ret  = H5Awrite(attr, H5T_NATIVE_CHAR, buf.data());
  ret  = H5Sclose(aid);
  ret  = H5Aclose(attr);
  if (ret<0)
    throw FatalErrorException(A_FUNCINFO,String("Can not write attribute ")+name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EnsightHdfDataWriter::
endWrite()
{
  info() << "ENSIGHT HDF END WRITE";
  std::ofstream ofile("a_hdf5");
  ofile << "SPECIAL HDF5 CASEFILE\n";
  ofile.width(10);
  Integer nb_time = m_times.size();
  ofile << nb_time << '\n';
  for( Integer i=0; i<nb_time; ++i )
    ofile << "hdf5_ensight_" << (i+1) << '\n';
  m_geometry_top_group.close();
  m_variable_top_group.close();
  m_file_id.close();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EnsightHdfDataWriter::
setMetaData(const String& meta_data)
{
  ARCANE_UNUSED(meta_data);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EnsightHdfDataWriter::
write(IVariable* var,IData* data)
{
  eItemKind item_kind = var->itemKind();
  if (item_kind!=IK_Cell){
    pwarning() << "Can only export cell variable (name=" << var->name() << ")";
    return;
  }
  if (var->dimension()!=1){
    pwarning() << "Can only export 1-dimension variable (name=" << var->name() << ")";
    return;
  }

  ++m_variable_index;

  {

    HGroup variable_group;
    String variable_name(String("variable_")+m_variable_index);
    variable_group.create(m_variable_top_group,variable_name.localstr());

    _addStringAttribute(variable_group,"varName",var->name().localstr(),80);
    _addIntegerAttribute(variable_group,"varType",2); //  SCALAR_HDFRW,
    _addStringAttribute(variable_group,"varDescription","None",80);
    _addIntegerAttribute(variable_group,"mappingType",1); // PER_ELEM_HDFRW

    RealUniqueArray global_min_values(9);
    global_min_values.fill(0.0);
    RealUniqueArray global_max_values(9);
    global_max_values.fill(0.0);

    Integer index = 1;
    for( ItemGroupCollection::Enumerator igroup(m_groups); ++igroup; ++index ){
      const ItemGroup& group = *igroup;
    
      info() << " VARIABLE INDEX=" << m_variable_index << " INDEX=" << index << " name=" << var->name();
      

      HGroup part_group;
      String part_name(String("var_part_")+index);
      info() << "CREATE PART GROUP " << part_name;
      part_group.create(variable_group,part_name.localstr());

      HGroup domain_group;
      String domain_name("var_domain_1");
      info() << "CREATE DOMAIN GROUP " << domain_name;
      domain_group.create(part_group,domain_name.localstr());
      
      info() << " DOMAIN GROUP " << variable_name << "/" << part_name << "/" << domain_name;

      RealUniqueArray min_values(9);
      min_values.fill(0.0);
      RealUniqueArray max_values(9);
      max_values.fill(0.0);
      _saveVariableOnGroup(var,data,group,min_values,max_values,domain_group);
      _addRealArrayAttribute(domain_group,"mins",min_values);
      _addRealArrayAttribute(domain_group,"maxs",max_values);

      _addRealArrayAttribute(part_group,"mins",min_values);
      _addRealArrayAttribute(part_group,"maxs",max_values);

      if (index==1){
        global_min_values.copy(min_values);
        global_max_values.copy(max_values);
      }
      else{
        for( Integer i=0; i<9; ++i ){
          if (min_values[i]<global_min_values[i])
            global_min_values[i] = min_values[i];
          if (max_values[i]>global_max_values[i])
            global_max_values[i] = max_values[i];
        }
      }
    }

    _addRealArrayAttribute(variable_group,"mins",global_min_values);
    _addRealArrayAttribute(variable_group,"maxs",global_max_values);

  }

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EnsightHdfDataWriter::
_saveVariableOnGroup(IVariable* var,IData* data,const ItemGroup& group,
                     RealArrayView min_values,
                     RealArrayView max_values,
                     HGroup& domain_group)
{
  Ref<ISerializedData> sdata(data->createSerializedDataRef(false));
  info() << "SAVE VARIABLE var=" << var->fullName() << " on group=" << group.name();
  if (sdata->baseDataType()!=DT_Real)
    ARCANE_FATAL("Bad datatype (only DT_Real is allowed)");

  Span<const Byte> sbuffer = sdata->constBytes();
  const Real* ptr = reinterpret_cast<const Real*>(sbuffer.data());
  Span<const Real> true_values(ptr,sdata->nbBaseElement());
  RealUniqueArray values;

  // Sauve les valeurs des éléments
  // ATTENTION: suppose uniquement des hexa8
  Real min_val = FloatInfo<Real>::maxValue();
  Real max_val = -min_val;
  {
    HDataset item_dataset;

    Int32UniqueArray item_node_ids;
    ENUMERATE_ITEM(iitem,group){
      Real value = true_values[iitem.itemLocalId()];
      values.add(value);
      if (value<min_val)
        min_val = value;
      if (value>max_val)
        max_val = value;
    }
    info() << "MIN =" << min_val << " MAX=" << max_val;
    min_values[0] = min_val;
    max_values[0] = max_val;
    Integer nb_value = values.size();
    hsize_t dims[1];
    dims[0] = nb_value;
    
    HSpace hspace;
    hspace.createSimple(1,dims);

    item_dataset.create(domain_group,"hexa8",H5T_NATIVE_FLOAT,hspace,H5P_DEFAULT);
    item_dataset.write(H5T_NATIVE_DOUBLE,values.data());

    _addIntegerAttribute(item_dataset,"layout",1); // INTERLACE_ORDER_1D
    _addIntegerAttribute(item_dataset,"dataType",0); // Indique une valeur flottante
  }

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Post-traitement au format Ensight Hdf.
 */
class EnsightHdfPostProcessor
: public ArcaneEnsightHdfPostProcessorObject
{
 public:
  EnsightHdfPostProcessor(const ServiceBuildInfo& sbi)
  : ArcaneEnsightHdfPostProcessorObject(sbi), m_writer(0)
  {
  }
  ~EnsightHdfPostProcessor()
  {
    delete m_writer;
  }

  virtual IDataWriter* dataWriter() { return m_writer; }
  virtual void notifyBeginWrite()
  {
    m_writer = new EnsightHdfDataWriter(mesh(),groups(),times());
  }
  virtual void notifyEndWrite()
  {
    delete m_writer;
    m_writer = 0;
  }
  virtual void close() {}

 private:

  IDataWriter* m_writer;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SUB_DOMAIN_FACTORY(EnsightHdfPostProcessor,
                                   IPostProcessorWriter,
																	 EnsightHdfPostProcessor);

ARCANE_REGISTER_SERVICE_ENSIGHTHDFPOSTPROCESSOR(EnsightHdfPostProcessor,
                                                EnsightHdfPostProcessor);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
