// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VTKHdfPostProcessor.cc                                      (C) 2000-2022 */
/*                                                                           */
/* Pos-traitement au format VTK HDF.                                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Collection.h"
#include "arcane/utils/Enumerator.h"
#include "arcane/utils/Iostream.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/CheckedConvert.h"

#include "arcane/PostProcessorWriterBase.h"

#include "arcane/std/Hdf5Utils.h"
#include "arcane/std/VTKHdfPostProcessor_axl.h"

#include "arcane/FactoryService.h"
#include "arcane/IDataWriter.h"
#include "arcane/IMesh.h"
#include "arcane/IMeshSubMeshTransition.h"
#include "arcane/IData.h"
#include "arcane/ISerializedData.h"
#include "arcane/IItemFamily.h"
#include "arcane/VariableCollection.h"

#include "arcane/IParallelMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
using namespace Hdf5Utils;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VTKHdfDataWriter
: public TraceAccessor
, public IDataWriter
{
 public:

  VTKHdfDataWriter(IMesh* mesh,ItemGroupCollection groups,
                   RealConstArrayView times);

 public:

  void beginWrite(const VariableCollection& vars) override;
  void endWrite() override;
  void setMetaData(const String& meta_data) override;
  void write(IVariable* var,IData* data) override;

 private:

  IMesh* m_mesh;
  
  //! Liste des groupes à sauver
  ItemGroupCollection m_groups;

  //! Liste des temps
  UniqueArray<Real> m_times;
  
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
  void _addInt64ArrayAttribute(Hid& hid,const char* name,Span<const Int64> values);
  void _addStringAttribute(Hid& hid,const char* name,const String& value,hsize_t len);
  void _saveGroup(const ItemGroup& group,HGroup& domain_group);
  void _saveVariableOnGroup(IVariable* var,IData* data,const ItemGroup& group,RealArrayView min_values,
                            RealArrayView max_values,HGroup& domain_group);

  template<typename DataType> void
  _writeDataSet1D(HGroup& group,const String& name,Span<const DataType> values);
  template<typename DataType> void
  _writeDataSet1D(HGroup& group,const String& name,Array<DataType>& values)
  {
    _writeDataSet1D(group,name,values.constSpan());
  }
  template<typename DataType> void
  _writeDataSet2D(HGroup& group,const String& name,Span2<const DataType> values);
  template<typename DataType> void
  _writeDataSet2D(HGroup& group,const String& name,Array2<DataType>& values)
  {
    _writeDataSet2D(group,name,values.constSpan());
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VTKHdfDataWriter::
VTKHdfDataWriter(IMesh* mesh,ItemGroupCollection groups,RealConstArrayView times)
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

void VTKHdfDataWriter::
beginWrite(const VariableCollection& vars)
{
  warning() << "L'implémentation du format 'VTKHdf' n'est pas encore opérationnelle";
  
  StringBuilder sb("vtk_hdf_");
  sb += m_time_index;
  sb += ".hdf";
  m_filename = sb.toString();

  m_variable_index = 0;

  info() << "ENSIGHT HDF BEGIN WRITE file=" << m_filename;

  H5open();

  m_file_id.openTruncate(m_filename);
  HGroup top_group;
  top_group.create(m_file_id,"VTKHDF");

  std::array<Int64,2> version = { 1, 0 };
  _addInt64ArrayAttribute(top_group,"Version",version);

  std::array<Int64,2> nb_rank_array = { 1, 0 };
  Span<const Int64> ranks { nb_rank_array };

  IParallelMng* pm =  m_mesh->parallelMng();
  const Int32 nb_rank = pm->commSize();
  if (nb_rank!=1)
    ARCANE_FATAL("Only sequential output is allowed");

  CellGroup all_cells = m_mesh->allCells();
  NodeGroup all_nodes = m_mesh->allNodes();

  const Int32 nb_cell = all_cells.size();
  const Int32 nb_node = all_nodes.size();

  // Pour les connectivités, la taille du tableau est égal
  // au nombre de mailes plus 1.
  UniqueArray<Int64> cells_connectivity;
  UniqueArray<Int64> cells_offset;
  UniqueArray<unsigned char> cells_type;
  const int VTK_HEXAHEDRON = 12;
  cells_offset.add(0);
  ENUMERATE_CELL(icell,all_cells){
    Cell cell = *icell;
    cells_type.add(VTK_HEXAHEDRON);
    for ( Node node : cell.nodes() )
      cells_connectivity.add(node.localId());
    cells_offset.add(cells_connectivity.size());
  }

  _writeDataSet1D(top_group,"Offsets",cells_offset);
  _writeDataSet1D(top_group,"Connectivity",cells_connectivity);
  _writeDataSet1D(top_group,"Types",cells_type);

  UniqueArray<Int64> nb_cell_by_ranks(nb_rank);
  nb_cell_by_ranks[0] = nb_cell;
  _writeDataSet1D(top_group,"NumberOfCells",nb_cell_by_ranks);
  
  UniqueArray<Int64> nb_node_by_ranks(nb_rank);
  nb_node_by_ranks[0] = nb_node;
  _writeDataSet1D(top_group,"NumberOfPoints",nb_node_by_ranks);

  UniqueArray<Int64> number_of_connectivity_ids(nb_rank);
  number_of_connectivity_ids[0] = cells_connectivity.size();
  _writeDataSet1D(top_group,"NumberOfConnectivityIds",number_of_connectivity_ids);

  VariableNodeReal3& nodes_coordinates(m_mesh->nodesCoordinates());
  UniqueArray2<Real> points;
  points.resize(nb_node,3);
  ENUMERATE_NODE(inode,all_nodes){
    //Node node = *inode;
    Int32 index = inode.index();
    Real3 pos = nodes_coordinates[inode];
    points[index][0] = pos.x;
    points[index][1] = pos.y;
    points[index][2] = pos.z;
  }
  _writeDataSet2D(top_group,"Points",points);

#if 0
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
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
namespace
{
template<typename DataType> class HDFTraits;

template<> class HDFTraits<Int64>
{
 public:
  static hid_t hdfType() { return H5T_NATIVE_INT64; }
};

template<> class HDFTraits<Int32>
{
 public:
  static hid_t hdfType() { return H5T_NATIVE_INT32; }
};

template<> class HDFTraits<double>
{
 public:
  static hid_t hdfType() { return H5T_NATIVE_DOUBLE; }
};

template<> class HDFTraits<unsigned char>
{
 public:
  static hid_t hdfType() { return H5T_NATIVE_UINT8; }
};

}

template<typename DataType> void VTKHdfDataWriter::
_writeDataSet1D(HGroup& group,const String& name,Span<const DataType> values)
{
  hsize_t dims[1];
  dims[0] = values.size();
  HSpace hspace;
  hspace.createSimple(1,dims);
  HDataset dataset;
  const hid_t hdf_type = HDFTraits<DataType>::hdfType();
  dataset.create(group,name.localstr(),hdf_type,hspace,H5P_DEFAULT);
  dataset.write(hdf_type,values.data());
}

template<typename DataType> void VTKHdfDataWriter::
_writeDataSet2D(HGroup& group,const String& name,Span2<const DataType> values)
{
  hsize_t dims[2];
  dims[0] = values.dim1Size();
  dims[1] = values.dim2Size();
  HSpace hspace;
  hspace.createSimple(2,dims);
  HDataset dataset;
  const hid_t hdf_type = HDFTraits<DataType>::hdfType();
  dataset.create(group,name.localstr(),hdf_type,hspace,H5P_DEFAULT);
  dataset.write(hdf_type,values.data());
}

void VTKHdfDataWriter::
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
      for( NodeEnumerator inode(item.nodes()); inode.hasNext(); ++inode ){
        node_local_indexes[inode.itemLocalId()] = 0;
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

    Int32 nb_rank = 1;

    hsize_t dims[1];
    dims[0] = nb_rank;
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
      Integer index = 0;
      for( NodeEnumerator inode(item.nodes()); inode.hasNext(); ++inode ){
        //const Node& node = *inode;
        item_node_ids.add(node_local_indexes[inode.itemLocalId()]);
#if 0
        info() << " CELL uid=" << item.uniqueId() << " id=" << inode.itemLocalId()
               << " id2=" << item.node(index).localId()
               << " node_uid=" << node.uniqueId();
#endif
        ++index;
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

void VTKHdfDataWriter::
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

void VTKHdfDataWriter::
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

void VTKHdfDataWriter::
_addInt64ArrayAttribute(Hid& hid,const char* name,Span<const Int64> values)
{
  hsize_t len = values.size();
  hid_t aid  = H5Screate_simple(1, &len, 0);
  hid_t attr = H5Acreate2(hid.id(),name, H5T_NATIVE_INT64, aid, H5P_DEFAULT,H5P_DEFAULT);
  if (attr<0)
    ARCANE_FATAL("Can not create attribute '{0}'",name);
  int ret  = H5Awrite(attr, H5T_NATIVE_INT64, values.data());
  ret  = H5Sclose(aid);
  ret  = H5Aclose(attr);
  if (ret<0)
    ARCANE_FATAL("Can not write attribute '{0}'",name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VTKHdfDataWriter::
_addRealArrayAttribute(Hid& hid,const char* name,RealConstArrayView values)
{
  hsize_t len = values.size();
  hid_t aid  = H5Screate_simple(1, &len, 0);
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

void VTKHdfDataWriter::
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

void VTKHdfDataWriter::
endWrite()
{
#if 0
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
#endif
  m_file_id.close();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VTKHdfDataWriter::
setMetaData(const String& meta_data)
{
  ARCANE_UNUSED(meta_data);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VTKHdfDataWriter::
write(IVariable* var,IData* data)
{
  return;

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

void VTKHdfDataWriter::
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
class VTKHdfPostProcessor
: public ArcaneVTKHdfPostProcessorObject
{
 public:

  explicit VTKHdfPostProcessor(const ServiceBuildInfo& sbi)
  : ArcaneVTKHdfPostProcessorObject(sbi)
  {
  }

  IDataWriter* dataWriter() override { return m_writer.get(); }
  void notifyBeginWrite() override
  {
    m_writer = std::make_unique<VTKHdfDataWriter>(mesh(), groups(), times());
  }
  void notifyEndWrite() override
  {
    m_writer = nullptr;
  }
  void close() override {}

 private:

  std::unique_ptr<IDataWriter> m_writer;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SUB_DOMAIN_FACTORY(VTKHdfPostProcessor,
                                   IPostProcessorWriter,
																	 VTKHdfPostProcessor);

ARCANE_REGISTER_SERVICE_VTKHDFPOSTPROCESSOR(VTKHdfPostProcessor,
                                            VTKHdfPostProcessor);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
