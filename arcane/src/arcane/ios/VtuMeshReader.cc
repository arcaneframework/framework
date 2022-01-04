// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VtuMeshReader.cc                                            (C) 2000-2013 */
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

#include "arcane/AbstractService.h"
#include "arcane/FactoryService.h"
#include "arcane/IMainFactory.h"
#include "arcane/IMeshReader.h"
#include "arcane/ISubDomain.h"
#include "arcane/IMesh.h"
#include "arcane/IMeshSubMeshTransition.h"
#include "arcane/IItemFamily.h"
#include "arcane/Item.h"
#include "arcane/ItemEnumerator.h"
#include "arcane/VariableTypes.h"
#include "arcane/IVariableAccessor.h" 
#include "arcane/IParallelMng.h"
#include "arcane/IIOMng.h"
#include "arcane/IXmlDocumentHolder.h"
#include "arcane/XmlNodeList.h"
#include "arcane/XmlNode.h"
#include "arcane/IMeshUtilities.h"
#include "arcane/IMeshWriter.h"
#include "arcane/BasicService.h"
#include "arcane/ItemPrinter.h"
#include "arcane/ServiceBuilder.h"

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


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Lecteur des fichiers de maillage aux format Vtk.
 */
class VtuMeshReader
: public AbstractService, public IMeshReader
{
 public:

  VtuMeshReader(const ServiceBuildInfo& sbi);

 public:

  virtual void build() {}

  bool allowExtension(const String& str);
		
  virtual eReturnType readMeshFromFile(IPrimaryMesh* mesh,const XmlNode& mesh_node,
                                       const String& file_name, const String& dir_name,bool use_internal_partition);
  eReturnType readMeshFromVtuFile(IMesh* mesh,const XmlNode& mesh_node,
                                  const String& file_name, const String& dir_name,bool use_internal_partition);

	ISubDomain* subDomain() { return m_sub_domain; }
	
	bool readGroupsFromFieldData(IMesh *mesh, vtkFieldData*, int);

 private:

  int vtkFileExtIdx;
  ISubDomain* m_sub_domain;
  eReturnType writeMeshToLemFile(IMesh* mesh, const String& filename,const String& dir_name);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SUB_DOMAIN_FACTORY(VtuMeshReader,IMeshReader,VtuNewMeshReader);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/****************************************************************************
 * Assignation des data types/file-type and particular file extension readers *
\****************************************************************************/
typedef struct{
  char *ext;
  IMeshReader::eReturnType (VtuMeshReader::*reader)(IMesh*,const XmlNode&,const String&,const String&,bool);
} vtkExtReader;

vtkExtReader vtkFileExtReader[]={
  {VTK_FILE_EXT_VTU, &VtuMeshReader::readMeshFromVtuFile},
  {NULL,NULL}
};


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VtuMeshReader::
VtuMeshReader(const ServiceBuildInfo& sbi)
: AbstractService(sbi)
, m_sub_domain(sbi.subDomain())
{
  vtkFileExtIdx=0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*****************************************************************
 * allowExtension
 *****************************************************************/
bool VtuMeshReader::
allowExtension(const String& str)
{
  //info() << "[allowExtension] Checking for file extension...";
  for(vtkFileExtIdx=0;vtkFileExtReader[vtkFileExtIdx].ext!=NULL;++vtkFileExtIdx){
    //info() << "Testing for '" << vtkFileExtReader[vtkFileExtIdx].ext << "'...";
    if (str == vtkFileExtReader[vtkFileExtIdx].ext){
      return true;
    }
  }
  //info()<<"Miss for all!";
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
                 const String& filename,const String& dir_name,
                 bool use_internal_partition)
{
  info() << "[readMeshFromFile] Forwarding to vtkFileExtReader[" << vtkFileExtIdx << "].reader";
  return (this->*vtkFileExtReader[vtkFileExtIdx].reader)(mesh, mesh_node, filename, dir_name, use_internal_partition);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*****************************************************************
 * writeMeshToLemFile
 *****************************************************************/
IMeshReader::eReturnType VtuMeshReader::
writeMeshToLemFile(IMesh* mesh, const String& filename,const String& dir_name)
{
  ISubDomain* sd = m_sub_domain;
  ScopedPtrT<IMeshWriter> mesh_writer(ServiceBuilder<IMeshWriter>::createInstance(sd,"Lima"));
  //FactoryT<IMeshWriter> mesh_writer_factory(sd->serviceMng());
  //mesh_writer = mesh_writer_factory.createInstance("Lima",true);
  if (!mesh_writer.get())
    pfatal() << "Mesh writer service selected not available";
  std::string fname = filename.localstr();
  fname += ".unf";
  if (mesh_writer->writeMeshToFile(mesh,fname) != RTOk)
    return RTError;
  return RTOk;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*****************************************************************
 * readGroupsFromFieldData
 *****************************************************************/
bool VtuMeshReader::
readGroupsFromFieldData(IMesh* mesh,vtkFieldData* allFieldData,int i)
{
	const std::string group_name = allFieldData->GetArrayName(i);
	vtkDataArray* iFieldData = allFieldData->GetArray(i);
	if (!iFieldData)
    return false;

	if (iFieldData->GetDataType() != VTK_LONG)
    return false;

  Integer nb_tuple = iFieldData->GetNumberOfTuples();
	info() << "[readGroupsFromFieldData] iFieldData->GetNumberOfTuples="<< nb_tuple;
  if (nb_tuple==0)
    return false;
	vtkLongArray* vtk_array =  vtkLongArray::SafeDownCast(iFieldData);

  // Le premier élément du tableau contient son type
	eItemKind kind_type = (eItemKind)(vtk_array->GetValue(0));
  IItemFamily* family = mesh->itemFamily(kind_type);
  Integer nb_item = nb_tuple - 1;
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
  for( Integer i=0; i<nb_item; ++i )
    if (local_ids[i]!=NULL_ITEM_LOCAL_ID)
      ids.add(local_ids[i]);

  info() << "Create group family=" << family->name() << " name=" << group_name << " ids=" << ids.size();
  family->createGroup(group_name,ids);
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*****************************************************************
 * readMeshFromVtuFile
 *****************************************************************/
IMeshReader::eReturnType VtuMeshReader::
readMeshFromVtuFile(IMesh* mesh, const XmlNode& mesh_node,
                    const String& filename,const String& dir_name,
                    bool use_internal_partition)
{
	bool itWasAnArcanProduction=true;
	IParallelMng* pm = mesh->parallelMng();
  bool is_parallel = pm->isParallel();
  Integer sid = pm->commRank();

	info() << "[readMeshFromVtuFile] Entering";
	vtkXMLUnstructuredGridReader *reader = vtkXMLUnstructuredGridReader::New();
	std::string fname = filename.localstr();
  reader->SetFileName(fname.c_str());
	if (!reader->CanReadFile(fname.c_str()))
    return RTError;
	reader->UpdateInformation();
	reader->Update();// Force reading
	vtkUnstructuredGrid *unstructuredGrid = reader->GetOutput();
  // Avec VTK 7, plus besoin du Update.
	//unstructuredGrid->Update();// La lecture effective du fichier n'a lieu qu'après l'appel à Update().
	int nbOfCells = unstructuredGrid->GetNumberOfCells();
	int nbOfNodes = unstructuredGrid->GetNumberOfPoints();


	/*******************************
   *Fetching Nodes UID & Cells UID *
   *******************************/
 	info() << "[readMeshFromVtuFile] ## Now Fetching Nodes Unique IDs ##";
	vtkPointData* allPointData=unstructuredGrid->GetPointData();			// Data associated to Points
	allPointData->Update();
	vtkDataArray* dataNodeArray = allPointData->GetArray("NodesUniqueIDs");
	vtkLongArray *nodesUidArray = nullptr;
	if (!dataNodeArray){
	 	info() << "[readMeshFromVtuFile] Could not be found, creating new one";
		nodesUidArray = vtkLongArray::New();
		for(Integer uid=0; uid<nbOfNodes; ++uid)
			nodesUidArray->InsertNextValue(uid);
		itWasAnArcanProduction=false;
	}
  else{
		if (dataNodeArray->GetDataType() != VTK_LONG)
      return RTError;
		nodesUidArray = vtkLongArray::SafeDownCast(dataNodeArray);
	}
 	info() << "[readMeshFromVtuFile] Fetched";	

 	info() << "[readMeshFromVtuFile] ## Now Fetching Cells Unique IDs ##";
	vtkCellData* allCellData=unstructuredGrid->GetCellData();					// Data associated to Points
	allCellData->Update();
	vtkDataArray* dataCellArray = allCellData->GetArray("CellsUniqueIDs");
	vtkLongArray *cellsUidArray = nullptr;
	if (!dataCellArray){
	 	info() << "[readMeshFromVtuFile] Could not be found, creating new one";
		cellsUidArray = vtkLongArray::New();
		for(Integer uid=0; uid<nbOfCells; ++uid)
			cellsUidArray->InsertNextValue(uid);
		itWasAnArcanProduction=false;
	}
  else{
		if (dataCellArray->GetDataType() != VTK_LONG)
      return RTError;
		cellsUidArray = vtkLongArray::SafeDownCast(dataCellArray);
	}

	// Tableau contenant les numéros des propriétaires des noeuds.
  // Ce tableau est optionnel et est utilisé par le partitionneur
  vtkDataArray* data_owner_array = allPointData->GetArray("NodesOwner");
	vtkIntArray* nodes_owner_array = 0;
	if (data_owner_array){
		if (data_owner_array->GetDataType() == VTK_INT){
      nodes_owner_array = vtkIntArray::SafeDownCast(data_owner_array);
    }
    else
      warning() << "NodesOwner array is present but has bad type (not Int32)";
  }
 	info() << "[readMeshFromVtuFile] Fetched";


	/************************
   * Fetch own nodes coords *
   ************************/
	info() << "[readMeshFromVtuFile] nbOfCells=" << nbOfCells << ", nbOfNodes=" << nbOfNodes;
  Real3UniqueArray coords(nbOfNodes);
	HashTableMapT<Int64,Real3> nodes_coords(nbOfNodes,true);
	vtkPoints* vtkAllPoints=unstructuredGrid->GetPoints();
	for(Integer i=0; i<nbOfNodes; ++i ){
		double xyz[3];
		vtkAllPoints->GetPoint(i,xyz);
		//info() << "x=" << xyz[0] << " y=" << xyz[1]<< " z=" << xyz[2] << ", nodUid=" << nodesUidArray->GetValue(i);
		coords[i] = Real3(xyz[0],xyz[1],xyz[2]);
		nodes_coords.nocheckAdd(nodesUidArray->GetValue(i),coords[i]);
  }

  // Create hash table for nodes owner.
	HashTableMapT<Int64,Int32> nodes_owner_map(nbOfNodes,true);
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
	Integer mesh_nb_cell_node = 0;
	for( Integer j=0, js=cells_filter.size(); j<js; ++j )
		mesh_nb_cell_node += unstructuredGrid->GetCell(j)->GetNumberOfPoints();
	info() << "Number of mesh_nb_cell_node = "<<mesh_nb_cell_node;
	
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
    case(VTK_QUAD):					arcItemType = IT_Quad4; 			break;
    case(VTK_TRIANGLE):				arcItemType = IT_Triangle3; 		break;
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
      case (3): arcItemType = IT_Triangle3; 		break;
      case (4): arcItemType = IT_Quad4; 			break;
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
      info() << "VTK_EMPTY_CELL n="<<cell->GetNumberOfPoints();
      break;
    case(VTK_TRIANGLE_STRIP):
      info() << "VTK_TRIANGLE_STRIP n="<<cell->GetNumberOfPoints();
      break;
    case(VTK_VOXEL):
      info() << "VTK_VOXEL n="<<cell->GetNumberOfPoints();
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
		int nNodes = cell->GetNumberOfPoints();// Return the number of points in the cell

		// First is cell's TYPE Stocke le type de la maille
		cells_infos[cells_infos_index++]=arcItemType;
		
		// Then comes its UniqueID Stocke le numéro unique de la maille
		Integer cell_indirect_id = cells_filter[i_cell];
		cells_infos[cells_infos_index++] = cellsUidArray->GetValue(cell_indirect_id);
		
		// And finally the Nodes' unique IDs
		vtkIdList* nodeIds=cell->GetPointIds();
		for(int iNode=0; iNode<nNodes; ++iNode){
			Integer localId = nodeIds->GetId(iNode);
			long uniqueUid=nodesUidArray->GetValue(localId);
			//info() << "working on localId=" << localId << ", uniqueUid=" << uniqueUid;
			cells_infos[cells_infos_index++] = uniqueUid;
		}	
	}
	
	
	/********************************
   * Setting Dimension & Allocating *
   ********************************/
	info() << "[readMeshFromVtuFile] ## Mesh 3D ##";
	PRIMARYMESH_CAST(mesh)->setDimension(3);
	info() << "[readMeshFromVtuFile] ## Allocating " <<  cells_filter.size() << " cells ##";
	PRIMARYMESH_CAST(mesh)->allocateCells(cells_filter.size(), cells_infos, false);


  // Positionne les propriétaires des noeuds à partir des groupes de noeuds
	ItemInternalList internalNodes(mesh->itemsInternal(IK_Node));
	info() << "[readMeshFromVtuFile] internalNodes.size()="<<internalNodes.size();
  if (nodes_owner_array){
    info() << "Set nodes owners from vtu file";
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
	info() << "[readMeshFromVtuFile] internalCells.size()="<<internalCells.size();
	for(Integer i=0, is=internalCells.size(); i<is; ++i)
    internalCells[i]->setOwner(sid,sid);
		

	/********************************************
   * Now finishing & preparing for ghost layout *
   ********************************************/
	info() << "[readMeshFromVtuFile] ## Ending with endAllocate ##";
	PRIMARYMESH_CAST(mesh)->endAllocate();
  if (is_parallel) {
    // mesh->setOwnersFromCells();
    mesh->utilities()->changeOwnersFromCells();
  }

	info() << "\n[readMeshFromVtuFile] ## Now dealing with ghost's layer ##";
	info() << "[readMeshFromVtuFile] mesh.nbNode=" <<mesh->nbNode() << " mesh.nbCell="<< mesh->nbCell();
 

	/***********************
   * Fetching Other Groups *
   ***********************/
	info() << "[readMeshFromVtuFile] ## Now Fetching Other Fields ##";
	vtkFieldData* allFieldData=unstructuredGrid->GetFieldData();
	int nbrOfArrays = allFieldData->GetNumberOfArrays();
	info() << "[readMeshFromVtuFile] nbrOfArrays = " << nbrOfArrays;
	for(int i=0;i<nbrOfArrays;++i){
		if (itWasAnArcanProduction==false) continue;
		info() << "[readMeshFromVtuFile] Focussing on \"" << allFieldData->GetArrayName(i) << "\" (i="<<i<<")";
		if (readGroupsFromFieldData(mesh, allFieldData, i) != true)
      return RTError;
	}


	/*******************
   * Now insert coords *
   *******************/
	info() << "[readMeshFromVtuFile] ##  Now insert coords ##";
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
	info() << "[readMeshFromVtuFile] RTOk";

  // TODO: regarder comment detruire automatiquement
  if (!dataNodeArray)
    nodesUidArray->Delete();
  if (!dataCellArray)
    cellsUidArray->Delete();

	return RTOk;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
