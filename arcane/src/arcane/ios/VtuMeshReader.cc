// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VtuMeshReader.cc                                            (C) 2000-2025 */
/*                                                                           */
/* Lecture/Ecriture d'un fichier au format VtuMeshReader.                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"
#include "arcane/utils/Iostream.h"
#include "arcane/utils/StdHeader.h"
#include "arcane/utils/HashTableMap.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/String.h"
#include "arcane/utils/IOException.h"
#include "arcane/utils/Collection.h"
#include "arcane/utils/Enumerator.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/Real3.h"

#include "arcane/core/AbstractService.h"
#include "arcane/core/FactoryService.h"
#include "arcane/core/IMainFactory.h"
#include "arcane/core/IMeshReader.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IMeshSubMeshTransition.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/Item.h"
#include "arcane/core/ItemEnumerator.h"
#include "arcane/core/VariableTypes.h"
#include "arcane/core/IVariableAccessor.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/IIOMng.h"
#include "arcane/core/IXmlDocumentHolder.h"
#include "arcane/core/XmlNodeList.h"
#include "arcane/core/XmlNode.h"
#include "arcane/core/IMeshUtilities.h"
#include "arcane/core/IMeshWriter.h"
#include "arcane/core/BasicService.h"
#include "arcane/core/ICaseMeshReader.h"
#include "arcane/core/IMeshBuilder.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/MeshKind.h"
#include "arcane/core/ServiceBuilder.h"

#include <vtkXMLUnstructuredGridReader.h>
#include <vtkUnstructuredGrid.h>
#include <vtkCell.h>
#include <vtkDataObject.h>
#include <vtkDataSet.h>
#include <vtkCellData.h>
#include <vtkPointData.h>
#include <vtkLongArray.h>
#include <vtkIntArray.h>


/**********************************************************************
 * Convention data types/file type and particular file extension parity *
 **********************************************************************/
#define VTK_FILE_EXT_VTI	"vti"	// Serial structured vtkImageData
#define VTK_FILE_EXT_VTP	"vtp"	// Serial UNstructured vtkPolyData
#define VTK_FILE_EXT_VTR	"vtr"	// Serial structured vtkRectilinearGrid
#define VTK_FILE_EXT_VTS	"vts"	// Serial structured vtkStructuredGrid
#define VTK_FILE_EXT_VTU	"vtu"	// Serial UNstructured vtkUnstructuredGrid
#define VTK_FILE_EXT_PVTI	"pvti"	// Parallel structured vtkImageData
#define VTK_FILE_EXT_PVTP	"pvtp"	// Parallel UNstructured vtkPolyData
#define VTK_FILE_EXT_PVTR	"pvtr"	// Parallel structured vtkRectilinearGrid
#define VTK_FILE_EXT_PVTS	"pvts"	// Parallel structured vtkStructuredGrid
#define VTK_FILE_EXT_PVTU	"pvtu"	// Parallel UNstructured vtkUnstructuredGrid

#if VTK_VERSION_NUMBER >= 90000
#define CURRENT_VTK_VERSION_LONG_TYPE VTK_LONG_LONG
#include <vtkLongLongArray.h>
using vtkLongArrayType = vtkLongLongArray;
#else
#define CURRENT_VTK_VERSION_LONG_TYPE VTK_LONG
using vtkLongArrayType = vtkLongArray;
#endif

#define VTK_FILE_EXT_VTPU	"vtpu"	// Parallel UNstructured vtkUnstructuredGrid
#define VTK_FILE_EXT_PVTPU	"pvtpu"	// Parallel UNstructured vtkUnstructuredGrid


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Lecteur des fichiers de maillage aux format Vtk.
 */
class VtuMeshReaderBase
{
 public:

  explicit VtuMeshReaderBase(ITraceMng* trace_mng);
  virtual ~VtuMeshReaderBase() = default;

 public:

  virtual void build() {}

  IMeshReader::eReturnType readMeshFromVtuFile(IMesh* mesh,
                                               const String& file_name, const String& dir_name, bool use_internal_partition);

	// ISubDomain* subDomain() { return m_trace_mng; }
	
	bool readGroupsFromFieldData(IMesh *mesh, vtkFieldData*, int);

 private:

  int vtkFileExtIdx;
  ITraceMng* m_trace_mng;
  // IMeshReader::eReturnType writeMeshToLemFile(IMesh* mesh, const String& filename,const String& dir_name);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/****************************************************************************
 * Assignation des data types/file-type and particular file extension readers *
\****************************************************************************/
typedef struct{
  char *ext;
  IMeshReader::eReturnType (VtuMeshReaderBase::*reader)(IMesh*,const String&,const String&,bool);
} vtkExtReader;

vtkExtReader vtkFileExtReader[]={
  {VTK_FILE_EXT_VTU, &VtuMeshReaderBase::readMeshFromVtuFile},
  {nullptr,nullptr}
};


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VtuMeshReaderBase::
VtuMeshReaderBase(ITraceMng* trace_mng) : m_trace_mng(trace_mng)
{
  vtkFileExtIdx=0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*****************************************************************
 * readGroupsFromFieldData
 *****************************************************************/
bool VtuMeshReaderBase::
readGroupsFromFieldData(IMesh* mesh,vtkFieldData* allFieldData,int i)
{
	const std::string group_name = allFieldData->GetArrayName(i);
	vtkDataArray* iFieldData = allFieldData->GetArray(i);
	if (!iFieldData)
    return false;

	if (iFieldData->GetDataType() != CURRENT_VTK_VERSION_LONG_TYPE)
    return false;

  auto nb_tuple = iFieldData->GetNumberOfTuples();
	m_trace_mng->info() << "[readGroupsFromFieldData] iFieldData->GetNumberOfTuples="<< nb_tuple;
  if (nb_tuple==0)
    return false;
	vtkLongArrayType* vtk_array =  vtkLongArrayType::SafeDownCast(iFieldData);

  // Le premier élément du tableau contient son type
	eItemKind kind_type = (eItemKind)(vtk_array->GetValue(0));
  IItemFamily* family = mesh->itemFamily(kind_type);
  auto nb_item = nb_tuple - 1;
  Int64UniqueArray unique_ids(nb_item);
  // Les éléments suivant contiennent les uniqueId() des entités du groupe.
  for( Integer z=0; z<nb_item; ++z )
    unique_ids[z] = vtk_array->GetValue(z+1);

  // Récupère le localId() correspondant.
  Int32UniqueArray local_ids(unique_ids.size());
  family->itemsUniqueIdToLocalId(local_ids,unique_ids,false);

  // Tous les entités ne sont pas forcément dans le maillage actuel et
  // il faut donc les filtrer.
  Int32UniqueArray ids;
  for( Integer index = 0; index < nb_item; ++index )
    if (local_ids[index]!=NULL_ITEM_LOCAL_ID)
      ids.add(local_ids[index]);

  m_trace_mng->info() << "Create group family=" << family->name() << " name=" << group_name << " ids=" << ids.size();
  family->createGroup(group_name,ids);
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*****************************************************************
 * readMeshFromVtuFile
 *****************************************************************/
IMeshReader::eReturnType VtuMeshReaderBase::
readMeshFromVtuFile(IMesh* mesh,
                    const String& file_name, const String& dir_name,
                    bool use_internal_partition)
{
  ARCANE_UNUSED(dir_name);
  ARCANE_UNUSED(use_internal_partition);
	bool itWasAnArcanProduction=true;
	IParallelMng* pm = mesh->parallelMng();
  bool is_parallel = pm->isParallel();
  Integer sid = pm->commRank();

	m_trace_mng->info() << "[readMeshFromVtuFile] Entering";
	vtkXMLUnstructuredGridReader *reader = vtkXMLUnstructuredGridReader::New();
	std::string fname = file_name.localstr();
  reader->SetFileName(fname.c_str());
	if (!reader->CanReadFile(fname.c_str()))
    return IMeshReader::RTError;
	reader->UpdateInformation();
	reader->Update();// Force reading
	vtkUnstructuredGrid *unstructuredGrid = reader->GetOutput();
  // Avec VTK 7, plus besoin du Update.
	//unstructuredGrid->Update();// La lecture effective du fichier n'a lieu qu'après l'appel à Update().
	auto nbOfCells = unstructuredGrid->GetNumberOfCells();
	auto nbOfNodes = unstructuredGrid->GetNumberOfPoints();


	/*******************************
   *Fetching Nodes UID & Cells UID *
   *******************************/
 	m_trace_mng->info() << "[readMeshFromVtuFile] ## Now Fetching Nodes Unique IDs ##";
	vtkPointData* allPointData=unstructuredGrid->GetPointData();			// Data associated to Points
	allPointData->Update();
	vtkDataArray* dataNodeArray = allPointData->GetArray("NodesUniqueIDs");
	vtkLongArrayType *nodesUidArray = nullptr;
  if (!dataNodeArray){
	 	m_trace_mng->info() << "[readMeshFromVtuFile] Could not be found, creating new one";
		nodesUidArray = vtkLongArrayType::New();
		for(Integer uid=0; uid<nbOfNodes; ++uid)
			nodesUidArray->InsertNextValue(uid);
		itWasAnArcanProduction=false;
	}
  else {
    m_trace_mng->info() << "dataNodeArray->GetDataType()" << dataNodeArray->GetDataType();
		if (dataNodeArray->GetDataType() != CURRENT_VTK_VERSION_LONG_TYPE)
      return IMeshReader::RTError;
		nodesUidArray = vtkLongArrayType::SafeDownCast(dataNodeArray);
	}
 	m_trace_mng->info() << "[readMeshFromVtuFile] Fetched";	

 	m_trace_mng->info() << "[readMeshFromVtuFile] ## Now Fetching Cells Unique IDs ##";
	vtkCellData* allCellData=unstructuredGrid->GetCellData();					// Data associated to Points
	allCellData->Update();
	vtkDataArray* dataCellArray = allCellData->GetArray("CellsUniqueIDs");
	vtkLongArrayType *cellsUidArray = nullptr;
	if (!dataCellArray){
	 	m_trace_mng->info() << "[readMeshFromVtuFile] Could not be found, creating new one";
		cellsUidArray = vtkLongArrayType::New();
		for(Integer uid=0; uid<nbOfCells; ++uid)
			cellsUidArray->InsertNextValue(uid);
		itWasAnArcanProduction=false;
	}
  else {
		if (dataCellArray->GetDataType() != CURRENT_VTK_VERSION_LONG_TYPE)
      return IMeshReader::RTError;
		cellsUidArray = vtkLongArrayType::SafeDownCast(dataCellArray);
	}

	// Tableau contenant les numéros des propriétaires des noeuds.
  // Ce tableau est optionnel et est utilisé par le partitionneur
  vtkDataArray* data_owner_array = allPointData->GetArray("NodesOwner");
	vtkIntArray* nodes_owner_array = nullptr;
	if (data_owner_array){
		if (data_owner_array->GetDataType() == VTK_INT) {
      nodes_owner_array = vtkIntArray::SafeDownCast(data_owner_array);
    }
    else {
      m_trace_mng->warning() << "NodesOwner array is present but has bad type (not Int32)";
    }
  }
 	m_trace_mng->info() << "[readMeshFromVtuFile] Fetched";


	/************************
   * Fetch own nodes coords *
   ************************/
	m_trace_mng->info() << "[readMeshFromVtuFile] nbOfCells=" << nbOfCells << ", nbOfNodes=" << nbOfNodes;
  Real3UniqueArray coords(nbOfNodes);
	HashTableMapT<Int64,Real3> nodes_coords(Integer(nbOfNodes),true);
	vtkPoints* vtkAllPoints=unstructuredGrid->GetPoints();
  // Get mesh dimension
  mesh->toPrimaryMesh()->setDimension(vtkAllPoints->GetData()->GetNumberOfComponents());
	for(vtkIdType i=0; i<nbOfNodes; ++i ){
		double xyz[3];
		vtkAllPoints->GetPoint(i,xyz);
		//m_trace_mng->info() << "x=" << xyz[0] << " y=" << xyz[1]<< " z=" << xyz[2] << ", nodUid=" << nodesUidArray->GetValue(i);
		coords[i] = Real3(xyz[0],xyz[1],xyz[2]);
		nodes_coords.nocheckAdd(nodesUidArray->GetValue(i),coords[i]);
  }

  // Create hash table for nodes owner.
	HashTableMapT<Int64,Int32> nodes_owner_map(Integer(nbOfNodes),true);
  if (nodes_owner_array){
    for(Integer i=0; i<nbOfNodes; ++i ){
      nodes_owner_map.nocheckAdd(nodesUidArray->GetValue(i),nodes_owner_array->GetValue(i));
    }
  }

	IntegerUniqueArray cells_filter;
	cells_filter.resize(nbOfCells);
	for( Integer i=0; i<nbOfCells; ++i )
		cells_filter[i] = i;
	
	// Calcul le nombre de mailles/noeuds
	vtkIdType mesh_nb_cell_node = 0;
	for( Integer j=0, js=cells_filter.size(); j<js; ++j )
		mesh_nb_cell_node += unstructuredGrid->GetCell(j)->GetNumberOfPoints();
	m_trace_mng->info() << "Number of mesh_nb_cell_node = "<<mesh_nb_cell_node;
	
	// Tableau contenant les infos aux mailles (voir IMesh::allocateMesh())
	Int64UniqueArray cells_infos(mesh_nb_cell_node+cells_filter.size()*2);
	Integer cells_infos_index = 0;

	/******************
   * Now insert CELLS *
   ******************/
	// For all cells that are discovered
	for( Integer i_cell=0, s_cell=cells_filter.size(); i_cell<s_cell; ++i_cell ){
		vtkIdType iVtkCell=i_cell;
		vtkCell *cell = unstructuredGrid->GetCell(iVtkCell);
		if (!cell->IsLinear())throw NotImplementedException(A_FUNCINFO);
		Integer arcItemType = IT_NullType;
		switch(cell->GetCellType()){
			// Linear cells
    case(VTK_TETRA):	 				arcItemType = IT_Tetraedron4;		break;
    case(VTK_HEXAHEDRON):			arcItemType = IT_Hexaedron8; 		break;
    case(VTK_PYRAMID):				arcItemType = IT_Pyramid5;			break;
			/* Others not yet implemented */
    case(VTK_WEDGE):					arcItemType = IT_Pentaedron6; 	break;
    case(VTK_PENTAGONAL_PRISM):	arcItemType = IT_Heptaedron10;	break;
    case(VTK_HEXAGONAL_PRISM):		arcItemType = IT_Octaedron12; 	break;
			/* 2D */
    case(VTK_QUAD):
		  if (mesh->dimension() == 2){ arcItemType = IT_Quad4;}
		  else if (mesh->dimension() == 3 && !mesh->meshKind().isMonoDimension()) {arcItemType = IT_Cell3D_Quad4;}
		  else {ARCANE_FATAL("VTK_QUAD is not supported in mono-dimension meshes of dimension {0}",mesh->dimension());}
		  break;
    case(VTK_TRIANGLE):
		  if (mesh->dimension() == 2) {arcItemType = IT_Triangle3;}
		  else if (mesh->dimension() == 3 && !mesh->meshKind().isMonoDimension()) {arcItemType = IT_Cell3D_Triangle3;}
		  else {ARCANE_FATAL("VTK_TRIANGLE is not supported in mono-dimension meshes of dimension {0}",mesh->dimension());}
		  break;
			// Cas du poly vertex à parser
    case(VTK_POLY_VERTEX):
      switch (cell->GetNumberOfPoints()){
      case (4): arcItemType = IT_Tetraedron4;	break;
      case (7): arcItemType = IT_HemiHexa7;		break;
      case (6): arcItemType = IT_HemiHexa6;		break;
      case (5): arcItemType = IT_HemiHexa5;		break;
      default:throw NotImplementedException(A_FUNCINFO);;
      }
      break;
    case(VTK_PIXEL):
      switch (cell->GetNumberOfPoints()){
      case (4): arcItemType = IT_Tetraedron4; 	break;
      default:throw NotImplementedException(A_FUNCINFO);;
      }
      break;
#ifndef NO_USER_WARNING
#warning IT_Vertex returns 0 nodes in ItemTypeMng.cc@42:type->setInfos(this,IT_Vertex,0,0,0);
#warning IT_Vertex vs IT_DualNode HACK
#endif /* NO_USER_WARNING */
    case(VTK_VERTEX):		arcItemType=IT_DualNode; 	break;
    case(VTK_POLYGON):
      switch (cell->GetNumberOfPoints()){
      case (3):
        if (mesh->dimension() == 2) {arcItemType = IT_Triangle3;}
        else if (mesh->dimension() == 3 && !mesh->meshKind().isMonoDimension()) {arcItemType = IT_Cell3D_Triangle3;}
        else {ARCANE_FATAL("VTK_TRIANGLE is not supported in mono-dimension meshes of dimension {0}",mesh->dimension());}
        break;
      case (4):
        if (mesh->dimension() == 2){ arcItemType = IT_Quad4;}
        else if (mesh->dimension() == 3 && !mesh->meshKind().isMonoDimension()) {arcItemType = IT_Cell3D_Quad4;}
        else {ARCANE_FATAL("VTK_QUAD is not supported in mono-dimension meshes of dimension {0}",mesh->dimension());}
        break;
      case (5): arcItemType = IT_Pentagon5; 		break;
      case (6): arcItemType = IT_Hexagon6; 		break;
      default:throw NotImplementedException(A_FUNCINFO);;
      }
      break;
    case(VTK_LINE):
    case(VTK_POLY_LINE):	
      switch (cell->GetNumberOfPoints()){
      case (2): arcItemType = IT_CellLine2; 	break;
      default:throw NotImplementedException(A_FUNCINFO);;
      }
    break;
    case(VTK_EMPTY_CELL):
      m_trace_mng->info() << "VTK_EMPTY_CELL n="<<cell->GetNumberOfPoints();
      break;
    case(VTK_TRIANGLE_STRIP):
      m_trace_mng->info() << "VTK_TRIANGLE_STRIP n="<<cell->GetNumberOfPoints();
      break;
    case(VTK_VOXEL):
      m_trace_mng->info() << "VTK_VOXEL n="<<cell->GetNumberOfPoints();
      switch (cell->GetNumberOfPoints()){
      case (3): arcItemType = IT_Triangle3; 	break;
      case (4): arcItemType = IT_Quad4; 		break;
      case (5): arcItemType = IT_Pentagon5; 	break;
      case (6): arcItemType = IT_Hexagon6; 	break;
      case (7): arcItemType = IT_HemiHexa7; 	break;
      case (8): arcItemType = IT_Hexaedron8;	break;
      default:throw NotImplementedException(A_FUNCINFO);;
      }
      break;
    default:throw NotImplementedException(A_FUNCINFO);;
		}
		auto nNodes = cell->GetNumberOfPoints();// Return the number of points in the cell

		// First is cell's TYPE Stocke le type de la maille
		cells_infos[cells_infos_index++]=arcItemType;
		
		// Then comes its UniqueID Stocke le numéro unique de la maille
		Integer cell_indirect_id = cells_filter[i_cell];
		cells_infos[cells_infos_index++] = cellsUidArray->GetValue(cell_indirect_id);
		
		// And finally the Nodes' unique IDs
		vtkIdList* nodeIds=cell->GetPointIds();
		for(int iNode=0; iNode<nNodes; ++iNode){
			auto localId = nodeIds->GetId(iNode);
			long uniqueUid=nodesUidArray->GetValue(localId);
			//info() << "working on localId=" << localId << ", uniqueUid=" << uniqueUid;
			cells_infos[cells_infos_index++] = uniqueUid;
		}	
	}
	
	
	/********************************
   * Setting Dimension & Allocating *
   ********************************/
	m_trace_mng->info() << "[readMeshFromVtuFile] ## Mesh 3D ##";
	PRIMARYMESH_CAST(mesh)->setDimension(3);
	m_trace_mng->info() << "[readMeshFromVtuFile] ## Allocating " <<  cells_filter.size() << " cells ##";
	PRIMARYMESH_CAST(mesh)->allocateCells(cells_filter.size(), cells_infos, false);


  // Positionne les propriétaires des noeuds à partir des groupes de noeuds
	ItemInternalList internalNodes(mesh->itemsInternal(IK_Node));
	m_trace_mng->info() << "[readMeshFromVtuFile] internalNodes.size()="<<internalNodes.size();
  if (nodes_owner_array){
    m_trace_mng->info() << "Set nodes owners from vtu file";
    for(Integer i=0, is=internalNodes.size(); i<is; ++i){
      ItemInternal* internal_node = internalNodes[i];
      Int32 true_owner = nodes_owner_map[internal_node->uniqueId()];
      internal_node->setOwner(true_owner,sid);
    }
  }
  else
    for(Integer i=0, is=internalNodes.size(); i<is; ++i)
      internalNodes[i]->setOwner(sid,sid);

	ItemInternalList internalCells(mesh->itemsInternal(IK_Cell));
	m_trace_mng->info() << "[readMeshFromVtuFile] internalCells.size()="<<internalCells.size();
	for(Integer i=0, is=internalCells.size(); i<is; ++i)
    internalCells[i]->setOwner(sid,sid);
		

	/********************************************
   * Now finishing & preparing for ghost layout *
   ********************************************/
	m_trace_mng->info() << "[readMeshFromVtuFile] ## Ending with endAllocate ##";
	PRIMARYMESH_CAST(mesh)->endAllocate();
  if (is_parallel) {
    // mesh->setOwnersFromCells();
    mesh->utilities()->changeOwnersFromCells();
  }

	m_trace_mng->info() << "\n[readMeshFromVtuFile] ## Now dealing with ghost's layer ##";
	m_trace_mng->info() << "[readMeshFromVtuFile] mesh.nbNode=" <<mesh->nbNode() << " mesh.nbCell="<< mesh->nbCell();
 

	/***********************
   * Fetching Other Groups *
   ***********************/
	m_trace_mng->info() << "[readMeshFromVtuFile] ## Now Fetching Other Fields ##";
	vtkFieldData* allFieldData=unstructuredGrid->GetFieldData();
	int nbrOfArrays = allFieldData->GetNumberOfArrays();
	m_trace_mng->info() << "[readMeshFromVtuFile] nbrOfArrays = " << nbrOfArrays;
	for(int i=0;i<nbrOfArrays;++i){
		if (itWasAnArcanProduction==false) continue;
		m_trace_mng->info() << "[readMeshFromVtuFile] Focussing on \"" << allFieldData->GetArrayName(i) << "\" (i="<<i<<")";
		if (readGroupsFromFieldData(mesh, allFieldData, i) != true)
      return IMeshReader::RTError;
	}


	/*******************
   * Now insert coords *
   *******************/
	m_trace_mng->info() << "[readMeshFromVtuFile] ##  Now insert coords ##";
	// Remplit la variable contenant les coordonnées des noeuds
	VariableNodeReal3& nodes_coord_var(PRIMARYMESH_CAST(mesh)->nodesCoordinates());
	ENUMERATE_NODE(inode,mesh->ownNodes()){
		nodes_coord_var[inode] = nodes_coords[inode->uniqueId()];
	}

	
	/****************************************
   * Synchronizing groups/variables & nodes *
   ****************************************/
  mesh->synchronizeGroupsAndVariables();

	/**************
	* Finishing up *
	**************/
	reader->Delete();
	m_trace_mng->info() << "[readMeshFromVtuFile] RTOk";

  // TODO: regarder comment detruire automatiquement
  if (!dataNodeArray)
    nodesUidArray->Delete();
  if (!dataCellArray)
    cellsUidArray->Delete();

	return IMeshReader::RTOk;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VtuMeshReader
: public AbstractService
, public IMeshReader
{
public:

  explicit VtuMeshReader(const ServiceBuildInfo& sbi)
  : AbstractService(sbi)
  , vtkFileExtIdx(-1)
  , m_sub_domain(sbi.subDomain())
  {}

public:

  bool allowExtension(const String& str) override;

  eReturnType readMeshFromFile(IPrimaryMesh* mesh,const XmlNode& mesh_node,
                                       const String& file_name, const String& dir_name,bool use_internal_partition) override;

private:

  int vtkFileExtIdx;
  ISubDomain* m_sub_domain;
  // eReturnType writeMeshToLemFile(IMesh* mesh, const String& filename,const String& dir_name);
};

/*****************************************************************
 * allowExtension
 *****************************************************************/
bool VtuMeshReader::
allowExtension(const String& str)
{
  //m_trace_mng->info() << "[allowExtension] Checking for file extension...";
  for(vtkFileExtIdx=0;vtkFileExtReader[vtkFileExtIdx].ext!=nullptr;++vtkFileExtIdx){
    //m_trace_mng->info() << "Testing for '" << vtkFileExtReader[vtkFileExtIdx].ext << "'...";
    if (str == vtkFileExtReader[vtkFileExtIdx].ext){
      return true;
    }
  }
  //m_trace_mng->info()<<"Miss for all!";
  // Sets our index in place for further service management
  vtkFileExtIdx=0;
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*****************************************************************
 * readMeshFromFile switch
 *****************************************************************/
IMeshReader::eReturnType VtuMeshReader::
readMeshFromFile(IPrimaryMesh* mesh,const XmlNode& mesh_node,
                 const String& file_name,const String& dir_name,
                 bool use_internal_partition)
{
  ARCANE_UNUSED(mesh_node);
  info() << "[readMeshFromFile] Forwarding to vtkFileExtReader[" << vtkFileExtIdx << "].reader";
  VtuMeshReaderBase vtu_mesh_reader{traceMng()};
  // return (vtu_mesh_reader.*vtkFileExtReader[vtkFileExtIdx].reader)(mesh, file_name, dir_name, use_internal_partition);
  return vtu_mesh_reader.readMeshFromVtuFile(mesh, file_name, dir_name, use_internal_partition);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


ARCANE_REGISTER_SUB_DOMAIN_FACTORY(VtuMeshReader,IMeshReader,VtuNewMeshReader);

ARCANE_REGISTER_SERVICE(VtuMeshReader,
                        ServiceProperty("VtuNewMeshReader", ST_SubDomain),
                        ARCANE_SERVICE_INTERFACE(IMeshReader));


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VtuCaseMeshReader
: public AbstractService
, public ICaseMeshReader
{
public:

  class Builder
  : public IMeshBuilder
  {
  public:

    explicit Builder(ITraceMng* tm, const CaseMeshReaderReadInfo& read_info)
    : m_trace_mng(tm)
    , m_read_info(read_info)
    {}

  public:

    void fillMeshBuildInfo(MeshBuildInfo& build_info) override
    {
      ARCANE_UNUSED(build_info);
    }
    void allocateMeshItems(IPrimaryMesh* pm) override
    {
      VtuMeshReaderBase vtu_service(pm->traceMng());
      String fname = m_read_info.fileName();
      m_trace_mng->info() << "Vtu Reader (ICaseMeshReader) file_name=" << fname;
      bool ret = vtu_service.readMeshFromVtuFile(pm, fname, m_read_info.directoryName(), m_read_info.isParallelRead());
      if (ret)
        ARCANE_FATAL("Can not read VTK File");
    }

  private:

    ITraceMng* m_trace_mng;
    CaseMeshReaderReadInfo m_read_info;
  };

public:

  explicit VtuCaseMeshReader(const ServiceBuildInfo& sbi)
  : AbstractService(sbi)
  {}

public:

  Ref<IMeshBuilder> createBuilder(const CaseMeshReaderReadInfo& read_info) const override
  {
    IMeshBuilder* builder = nullptr;
    if (read_info.format() == "vtu")
      builder = new Builder(traceMng(), read_info);
    return makeRef(builder);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(VtuCaseMeshReader,
                        ServiceProperty("VtuCaseMeshReader", ST_SubDomain),
                        ARCANE_SERVICE_INTERFACE(ICaseMeshReader));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
