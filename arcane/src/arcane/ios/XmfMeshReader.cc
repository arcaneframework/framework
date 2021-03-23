// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* XmfMeshReader.cc                                            (C) 2000-2010 */
/*                                                                           */
/* Lecture/Ecriture d'un fichier au format XMF.				                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"
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
#include "arcane/IPrimaryMesh.h"
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

#define XDMF_USE_ANSI_STDLIB
#include "vtkxdmf2/XdmfArray.h"
#include "vtkxdmf2/XdmfAttribute.h"
#include "vtkxdmf2/XdmfDOM.h"
#include "vtkxdmf2/XdmfDataDesc.h"
#include "vtkxdmf2/XdmfDataItem.h"
#include "vtkxdmf2/XdmfGrid.h"
#include "vtkxdmf2/XdmfTopology.h"
#include "vtkxdmf2/XdmfGeometry.h"
#include "vtkxdmf2/XdmfTime.h"

/*#define H5_USE_16_API
#include <hdf5.h>*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
ARCANE_BEGIN_NAMESPACE

using namespace xdmf2;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
 
/*!
 * \brief Lecteur des fichiers de maillage aux format msh.
 */
class XmfMeshReader: public AbstractService, public IMeshReader{
	public:
		XmfMeshReader(const ServiceBuildInfo& sbi);

	public:
		virtual void build() {}

		bool allowExtension(const String& str){return (str=="xmf");}
		
		virtual eReturnType readMeshFromFile(IPrimaryMesh* mesh,const XmlNode& mesh_node,
                                         const String& file_name, const String& dir_name,bool use_internal_partition);

		ISubDomain* subDomain() { return m_sub_domain; }
	
	private:
		ISubDomain* m_sub_domain;
  eReturnType _readTopology(XdmfTopology*, XdmfInt64&, Int32Array&, Int32Array&, Int32Array&, SharedArray<Int32>);
  eReturnType _readMixedTopology(XdmfTopology*, XdmfInt64, Int32Array&, Int32Array&, Int32Array&, SharedArray<Int32>);
};


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
ARCANE_REGISTER_SUB_DOMAIN_FACTORY(XmfMeshReader, IMeshReader, XmfNewMeshReader);
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

XmfMeshReader::XmfMeshReader(const ServiceBuildInfo& sbi):AbstractService(sbi), m_sub_domain(sbi.subDomain()){}



/*****************************************************************************\
* [_addThisConnectivity]																		*
\*****************************************************************************/
#define _addThisConnectivity(nNodes, arcType)\
	cells_type.add(arcType);\
	cells_nb_node.add(nNodes);\
	for (XdmfInt32 z=0; z<nNodes; ++z)\
		cells_connectivity.add(nodesUidArray[xdmfConnectivity->GetValueAsInt32(i++)]);



/*****************************************************************************\
* [_readMixedTopology]																			*
\*****************************************************************************/
IMeshReader::eReturnType XmfMeshReader::
_readMixedTopology(XdmfTopology *Topology,
                   XdmfInt64 nb_elements,
                   Int32Array& cells_nb_node,
                   Int32Array& cells_connectivity,
                   Int32Array& cells_type,
                   SharedArray<Int32> nodesUidArray)
{
	info() << "[_readMixedTopology] Entering";
	XdmfArray *xdmfConnectivity=Topology->GetConnectivity(NULL, XDMF_TRUE);
	XdmfInt32 numType=xdmfConnectivity->GetNumberType();
	info() << "xdmfConnectivity.CoreLength=" << xdmfConnectivity->GetCoreLength()<< " numType=" << numType;
	if (numType != XDMF_INT32_TYPE)
    throw NotSupportedException(A_FUNCINFO, "Not supported connectivity num type");
	XdmfInt32 nNodesForThisElement;
	XdmfInt64 iElementsScaned=0;
	for(XdmfInt64 i=0; iElementsScaned < nb_elements; iElementsScaned++){
	  // info() << "[_readMixedTopology] scan=" << iElementsScaned << "/" << nb_elements;
	  switch (xdmfConnectivity->GetValueAsInt32(i++)){
	  case (XDMF_POLYVERTEX):		// 0x1
		 nNodesForThisElement=xdmfConnectivity->GetValueAsInt32(i++);
		 // info() << "[_readMixedTopology] XDMF_POLYVERTEX(" << nNodesForThisElement << ")";
		 switch (nNodesForThisElement){
		 case (1): _addThisConnectivity(1, IT_Vertex); break;
		 case (12):	_addThisConnectivity(12, IT_Octaedron12); break;
     case (14): _addThisConnectivity(14, IT_Enneedron14); break;
     case (16): _addThisConnectivity(16, IT_Decaedron16); break;
		 case (7): _addThisConnectivity(7, IT_HemiHexa7); break;
		 case (6): _addThisConnectivity(6, IT_HemiHexa6); break;
       // GG: je commente ces deux cas car dans la version de Oct2015 cela ne compile
       // pas car IT_AntiWedgetLeft6 vaut 16 et donc la même valeur dans le switch
       // que pour le IT_Decaedron16. De toute facon, je pense que le case avant ne fonctionnait
       // pas car la valeur pour 6 noeuds est déjà prise par IT_HemiHexa6.
       //case (IT_AntiWedgeLeft6): _addThisConnectivity(6, IT_AntiWedgeLeft6); break;
       //case (IT_AntiWedgeRight6): _addThisConnectivity(6, IT_AntiWedgeRight6); break;
		 case (5): _addThisConnectivity(5, IT_HemiHexa5); break;
		 case (55): _addThisConnectivity(5, IT_DiTetra5); break;
		 default:
       throw FatalErrorException(A_FUNCINFO, "XDMF_POLYVERTEX with unknown number of nodes");
		 }
		 break;
	  case (XDMF_POLYLINE):		// 0x2
		 nNodesForThisElement=xdmfConnectivity->GetValueAsInt32(i++);
		 // info() << "[_readMixedTopology] XDMF_POLYLINE("<<nNodesForThisElement<<")";
		 switch (nNodesForThisElement){
		 case (2):	_addThisConnectivity(2, IT_Line2); break;
		 default: throw FatalErrorException(A_FUNCINFO, "XDMF_POLYLINE with unknown number of nodes");
		 }
		 break;
	  case (XDMF_POLYGON):			// 0x3
		 nNodesForThisElement=xdmfConnectivity->GetValueAsInt32(i++);
		 // info() << "[_readMixedTopology] XDMF_POLYGON("<<nNodesForThisElement<<")";
		 switch (nNodesForThisElement){
		 case (5):	_addThisConnectivity(5, IT_Pentagon5); break;
		 case (6):	_addThisConnectivity(6, IT_Hexagon6); break;
		 default: throw FatalErrorException(A_FUNCINFO, "XDMF_POLYGON with unknown number of nodes");
		 }
		 break;
	  case (XDMF_TRI):_addThisConnectivity(3, IT_Triangle3); break;// 0x4
	  case (XDMF_QUAD):_addThisConnectivity(4, IT_Quad4); break;// 0x5
	  case (XDMF_TET):_addThisConnectivity(4, IT_Tetraedron4); break;// 0x6
	  case (XDMF_PYRAMID):_addThisConnectivity(5, IT_Pyramid5); break;// 0x7
	  case (XDMF_WEDGE ):_addThisConnectivity(6, IT_Pentaedron6); break;// 0x8
	  case (XDMF_HEX):_addThisConnectivity(8, IT_Hexaedron8); break;// 0x9
	  case (XDMF_TET_10):_addThisConnectivity(10, IT_Heptaedron10); break;// 0x0026
	  case (XDMF_EDGE_3):
	  case (XDMF_TRI_6):
	  case (XDMF_QUAD_8):
	  case (XDMF_PYRAMID_13):
	  case (XDMF_WEDGE_15):
	  case (XDMF_HEX_20):
	  default: throw NotSupportedException(A_FUNCINFO, "Not supported topology type in a mixed one");
	  }
	}
//	delete  xdmfConnectivity;
	info() << "[_readMixedTopology] Done";
	return RTOk;
}


/*****************************************************************************\
* [_readTopology]																					*
   XML Element : Topology
   XML Attribute : Name = Any String
   XML Attribute : TopologyType = Polyvertex | Polyline | Polygon |
                                  Triangle | Quadrilateral | Tetrahedron | Pyramid| Wedge | Hexahedron |
                                  Edge_3 | Triagle_6 | Quadrilateral_8 | Tetrahedron_10 | Pyramid_13 |
                                  Wedge_15 | Hexahedron_20 |
                                  Mixed |
                                  2DSMesh | 2DRectMesh | 2DCoRectMesh |
                                  3DSMesh | 3DRectMesh | 3DCoRectMesh
   XML Attribute : NumberOfElements = Number of Cells
   XML Attribute : NodesPerElement = # (Only Important for Polyvertex, Polygon and Polyline)
   XML Attribute : Order = Order of Nodes if not Default
   XML BaseOffset: Offset if not 0
\*****************************************************************************/
IMeshReader::eReturnType XmfMeshReader::
_readTopology(XdmfTopology *Topology,
              XdmfInt64& nb_elements,
              Int32Array& cells_nb_node,
              Int32Array& cells_connectivity,
              Int32Array& cells_type,
              SharedArray<Int32> nodesUidArray)
{
	info() << "[_readTopology] Entering";
	if (!Topology) throw FatalErrorException(A_FUNCINFO, "Null topology");
	if (Topology->UpdateInformation() != XDMF_SUCCESS) throw FatalErrorException(A_FUNCINFO, "Error in UpdateInformation");
	if (Topology->Update() != XDMF_SUCCESS) throw FatalErrorException(A_FUNCINFO, "Error in Update");
	nb_elements = Topology->GetNumberOfElements();
	info() << "\tHave found a " << Topology->GetClassAsString() << " "
			 << Topology->GetTopologyTypeAsString() << " topology with " << nb_elements << " elements";
	switch (Topology->GetTopologyType()){
		case (XDMF_2DSMESH):			// 0x0100
		case (XDMF_2DRECTMESH):		// 0x0101
		case (XDMF_2DCORECTMESH):	// 0x0102
		case (XDMF_3DRECTMESH):		// 0x1101
		case (XDMF_3DCORECTMESH):	// 0x1102
		case (XDMF_3DSMESH):{		//	0x1100
		// Try to do something with these kind of structured topologies
			if ((nb_elements%3)==0){
				for(XdmfInt64 i=0; i<nb_elements/3; i+=3){
					cells_type.add(IT_Triangle3);
					cells_nb_node.add(3);
					for (Integer z=0; z<3; ++z)
						cells_connectivity.add(i+z);
				}
			}else if ((nb_elements%4)==0){
				for(XdmfInt64 i=0; i<nb_elements/4; i+=4){
					cells_type.add(IT_Quad4);
					cells_nb_node.add(4);
					for (Integer z=0; z<4; ++z)
						cells_connectivity.add(i+z);
				}
			}else throw FatalErrorException(A_FUNCINFO, "Could not match XDMF_3DSMESH with a known mesh");
			return RTOk;
		}
		// Otherwise, read a mixed topology should do the trick
		case (XDMF_POLYVERTEX):		// 0x1
		case (XDMF_POLYLINE):		// 0x2
		case (XDMF_POLYGON):			// 0x3
		case (XDMF_TRI):				// 0x4
		case (XDMF_QUAD):				// 0x5
		case (XDMF_TET):				// 0x6
		case (XDMF_PYRAMID):			// 0x7
		case (XDMF_WEDGE ):			// 0x8
		case (XDMF_HEX):				// 0x9
		case (XDMF_EDGE_3):			// 0x0022
		case (XDMF_TRI_6):			// 0x0024
		case (XDMF_QUAD_8):			// 0x0025
		case (XDMF_TET_10):			// 0x0026
		case (XDMF_PYRAMID_13):		// 0x0027
		case (XDMF_WEDGE_15):		// 0x0028
		case (XDMF_HEX_20):			// 0x0029
		case (XDMF_MIXED):			// 0x0070
		  if (_readMixedTopology(Topology, nb_elements, cells_nb_node, cells_connectivity, cells_type, nodesUidArray) != RTOk)
				throw FatalErrorException(A_FUNCINFO, "Error in _readMixedTopology");
			break;
		case (XDMF_NOTOPOLOGY):		// 0x0
		default: throw NotSupportedException(A_FUNCINFO, "Not supported topology type");
	}
//	Topology->Release();
//	delete Topology;
	info() << "[_readTopology] Release & Done";
	return RTOk;
}



/*****************************************************************************
 [readMeshFromFile]
The organization of XDMF begins with the Xdmf element. So that parsers can 
distinguish from previous versions of XDMF, there exists a Version attribute 
(currently at 2.0).

Xdmf elements contain one or more Domain elements (computational 
domain). There is seldom motivation to have more than one Domain. 

A Domain can have one or more Grid elements.

Each Grid contains a Topology, Geometry, and zero or more Attribute elements.

Topology specifies the connectivity of the grid while
Geometry specifies the location of the grid nodes.
*****************************************************************************/
IMeshReader::eReturnType XmfMeshReader::
readMeshFromFile(IPrimaryMesh* mesh,const XmlNode& mesh_node,
				       const String& file_name, const String& dir_name,bool use_internal_partition)
{
	IParallelMng* pm = mesh->parallelMng();
	bool is_parallel = pm->isParallel();
	Integer sid = pm->commRank();
	bool itWasAnArcanProduction=true;
	
	info() << "[readMeshFromFile] Entering";
	XdmfDOM *DOM = new XdmfDOM();

	// Parse the XML File
	if (DOM->SetInputFileName(file_name.localstr()) != XDMF_SUCCESS) throw FatalErrorException(A_FUNCINFO, "SetInputFileName");
	if (DOM->Parse() != XDMF_SUCCESS) throw FatalErrorException(A_FUNCINFO, "Parse");
	XdmfXmlNode XdmfRoot = DOM->GetTree();

	// Version verification
	if (strcmp(DOM->Get(XdmfRoot, "Version"), "2.0")<0) throw NotSupportedException(A_FUNCINFO, "Not supported Xdmf-file version");
	
	// Now scan all of its children
	XdmfInt64 nRootChildren=DOM->GetNumberOfChildren(XdmfRoot);
	info() << "GetNumberOfChildren=" << nRootChildren;

	//Xdmf elements contain one or more Domain elements
	XdmfInt32 nDomains= DOM->FindNumberOfElements("Domain");
	info() << "nDomains=" << nDomains;
	

	for(XdmfInt64 iDomain=0; iDomain < nDomains; ++iDomain){
		XdmfXmlNode foundDomain=DOM->FindElement("Domain", iDomain, XdmfRoot, 0);
		if (foundDomain != NULL) info() << "Have found domain" << iDomain;
		
		// A Domain can have one or more Grid elements.
		XdmfInt32 nGrids= DOM->FindNumberOfElements("Grid", foundDomain);
		info() << "nGrids=" << nGrids;
		for(XdmfInt32 iGrid=0; iGrid < nGrids; ++iGrid){
			/*****************************
			 Looking for the domain's Grid
			******************************/
			XdmfXmlNode foundGrid=DOM->FindElement("Grid", iGrid, foundDomain, 0);
			if (foundGrid == NULL) throw FatalErrorException(A_FUNCINFO, "Grid not found for domain");
			XdmfGrid *Grid = new XdmfGrid();
			Grid->SetDOM(DOM);
			Grid->SetElement(foundGrid);
			Grid->UpdateInformation();
			info() << "Have found a " << Grid->GetGridTypeAsString() << " grid";
			if (Grid->GetGridType() != XDMF_GRID_UNIFORM) throw NotSupportedException(A_FUNCINFO, "Not supported GRID type");
			
			/*****************************************************
			 Looking for XML attribute which Name="CellsUniqueIDs"
			******************************************************/
			XdmfXmlNode cellsUniqueIDsXmlNode = DOM->FindElementByAttribute ("Name", "CellsUniqueIDs", 0, foundGrid);
			Int32UniqueArray cellsUidArray;
			if (cellsUniqueIDsXmlNode) {
			  info() << "[XmfMeshReader] cellsUidArray were found";
			  XdmfAttribute *attribute = new XdmfAttribute();
			  attribute->SetDOM(DOM);
			  attribute->SetElement(cellsUniqueIDsXmlNode);
			  attribute->UpdateInformation();
			  attribute->Update();
			  XdmfArray *xmfGroup = attribute->GetValues();
			  info() << attribute->GetName() << "(" << attribute->GetAttributeTypeAsString() << ", "
						<< attribute->GetAttributeCenterAsString() << ", " << xmfGroup->GetNumberOfElements() <<")";
			  XdmfInt64 nb_item = xmfGroup->GetNumberOfElements();
			  for(XdmfInt64 uid=0; uid<nb_item; ++uid)
				 cellsUidArray.add(xmfGroup->GetValueAsInt32(uid));
			}else itWasAnArcanProduction=false;
			
			/*****************************************************
			 Looking for XML attribute which Name="NodesUniqueIDs"
			******************************************************/
			XdmfXmlNode nodesUniqueIDsXmlNode = DOM->FindElementByAttribute ("Name", "NodesUniqueIDs", 0, foundGrid);
			SharedArray<Int32> nodesUidArray;
			if (nodesUniqueIDsXmlNode) {
			  info() << "[XmfMeshReader] nodesUidArray were found";
			  XdmfAttribute *attribute = new XdmfAttribute();
			  attribute->SetDOM(DOM);
			  attribute->SetElement(nodesUniqueIDsXmlNode);
			  attribute->UpdateInformation();
			  attribute->Update();
			  XdmfArray *xmfGroup = attribute->GetValues();
			  info() << attribute->GetName() << "(" << attribute->GetAttributeTypeAsString() << ", "
						<< attribute->GetAttributeCenterAsString() << ", " << xmfGroup->GetNumberOfElements() <<")";
			  XdmfInt64 nb_item = xmfGroup->GetNumberOfElements();
			  for(XdmfInt64 uid=0; uid<nb_item; ++uid)
				 nodesUidArray.add(xmfGroup->GetValueAsInt32(uid));
			}
      else itWasAnArcanProduction=false;

			/*****************************************************
			 Looking for XML attribute which Name="NodesOwner"
			******************************************************/
			XdmfXmlNode nodesOwnerXmlNode = DOM->FindElementByAttribute ("Name", "NodesOwner", 0, foundGrid);
			Int32UniqueArray nodesOwnerArray;
			if (nodesOwnerXmlNode) {
			  info() << "[XmfMeshReader] nodesOwnerArray were found";
			  XdmfAttribute *attribute = new XdmfAttribute();
			  attribute->SetDOM(DOM);
			  attribute->SetElement(nodesOwnerXmlNode);
			  attribute->UpdateInformation();
			  attribute->Update();
			  XdmfArray *xmfGroup = attribute->GetValues();
			  info() << attribute->GetName() << "(" << attribute->GetAttributeTypeAsString() << ", "
						<< attribute->GetAttributeCenterAsString() << ", " << xmfGroup->GetNumberOfElements() <<")";
			  XdmfInt64 nb_item = xmfGroup->GetNumberOfElements();
			  for(XdmfInt64 uid=0; uid<nb_item; ++uid)
				 nodesOwnerArray.add(xmfGroup->GetValueAsInt32(uid));
			}
      else
        itWasAnArcanProduction=false;

			/******************************
			 Each Grid contains a Geometry,
			*******************************/
			XdmfGeometry* xmfGometry = Grid->GetGeometry();
			if (!xmfGometry)
        ARCANE_FATAL("No xmfGeometry");
      info() << "\tHave found a " << xmfGometry->GetGeometryTypeAsString() << " geometry";
      if (xmfGometry->GetGeometryType() != XDMF_GEOMETRY_XYZ)
        throw NotSupportedException(A_FUNCINFO, "Not supported geometry type");
      xmfGometry->UpdateInformation();
      xmfGometry->Update();
      HashTableMapT<Int64,Real3> nodes_coords(xmfGometry->GetNumberOfPoints(),true);
			XdmfInt64 nbOfNodes = xmfGometry->GetNumberOfPoints();

			/****************************************************
			 If it is not ours, just fake the indirection process 
			*****************************************************/
			if (!itWasAnArcanProduction){
			  info() << "If it is not ours, just fake the indirection process";
			  for(Integer uid=0; uid<nbOfNodes; ++uid){
				 nodesUidArray.add(uid);
				 cellsUidArray.add(uid);
				 nodesOwnerArray.add(uid);
			  }
			}
			
			XdmfArray *xdmfPoints =  xmfGometry->GetPoints(XDMF_TRUE);// true to create the array
			XdmfInt32 numType=xdmfPoints->GetNumberType();
			if (numType!=XDMF_FLOAT32_TYPE) throw NotSupportedException(A_FUNCINFO, "Not supported geometry number type");
			XdmfInt64 iNode=0;
			for(XdmfInt64 i=0; iNode<nbOfNodes; iNode++, i+=3){
			  Real3 coords=Real3(xdmfPoints->GetValueAsFloat32(i),
										xdmfPoints->GetValueAsFloat32(i+1),
										xdmfPoints->GetValueAsFloat32(i+2));
			  nodes_coords.nocheckAdd(nodesUidArray[iNode], coords);
			}

			
			/******************************
		    Each Grid contains a Topology,
			*******************************/
			XdmfInt64 nb_elements;
			Int32UniqueArray cells_nb_node;
			Int32UniqueArray cells_connectivity;
			Int32UniqueArray cells_type;
			XdmfTopology *topology=Grid->GetTopology();
			if (_readTopology(topology, nb_elements, cells_nb_node, cells_connectivity, cells_type, nodesUidArray) != RTOk)
				throw IOException("XmfMeshReader", "_readTopology error");
			
			// Create hash table for nodes owner.
			HashTableMapT<Int64,Int32> nodes_owner_map(nbOfNodes,true);
			if (nodesOwnerXmlNode && itWasAnArcanProduction){
			  info() << "[_XmfMeshReader] Create hash table for nodes owner";
			  for(Integer i=0; i<nbOfNodes; ++i ){
				 //info() << nodesUidArray[i] <<":"<<nodesOwnerArray[i];
				 nodes_owner_map.nocheckAdd(nodesUidArray[i], nodesOwnerArray[i]);
			  }
			}

			/****************************
			 * Now building cells_infos *
			 ****************************/
			Int64UniqueArray cells_infos;
			info() << "[_XmfMeshReader]  Création des mailles, nb_cell=" << nb_elements << " cells_type.size=" << cells_type.size();
			Integer connectivity_index = 0;
			for(Integer i=0; i<cells_type.size(); ++i ){
			  cells_infos.add(cells_type[i]);
			  cells_infos.add(cellsUidArray[i]);
			  for (Integer z=0; z<cells_nb_node[i]; ++z )
				 cells_infos.add(cells_connectivity[connectivity_index+z]);  
			  connectivity_index += cells_nb_node[i];
			}
			
			/********************************
			 * Setting Dimension & Allocating *
			 ********************************/
			info() << "[XmfMeshReader] ## Mesh 3D ##";
			mesh->setDimension(3);
			info() << "[XmfMeshReader] ## Allocating ##";
			mesh->allocateCells(cells_type.size(), cells_infos, false);

			/**********************************************************************
          Positionne les propriétaires des noeuds à partir des groupes de noeuds
			***********************************************************************/
			ItemInternalList internalNodes(mesh->itemsInternal(IK_Node));
			info() << "[XmfMeshReader] internalNodes.size()="<<internalNodes.size();
			if (nodesOwnerXmlNode && itWasAnArcanProduction){
			  info() << "[XmfMeshReader] Setting nodes owners from xmf file";
			  for(Integer i=0, is=internalNodes.size(); i<is; ++i){
				 ItemInternal* internal_node = internalNodes[i];
				 //info() << "[XmfMeshReader] "<<internal_node->uniqueId()<<":"<<nodes_owner_map[internal_node->uniqueId()];
				 Int32 true_owner = nodes_owner_map[internal_node->uniqueId()];
				 internal_node->setOwner(true_owner,sid);
			  }
			}
			else{
			  for(Integer i=0, is=internalNodes.size(); i<is; ++i)
				 internalNodes[i]->setOwner(sid,sid);
			}
			ItemInternalList internalCells(mesh->itemsInternal(IK_Cell));
			info() << "[XmfMeshReader] internalCells.size()="<<internalCells.size();
			for(Integer i=0, is=internalCells.size(); i<is; ++i)
			  internalCells[i]->setOwner(sid,sid);
			
			/********************************************
			 * Now finishing & preparing for ghost layout *
			 ********************************************/
			info() << "[XmfMeshReader] ## Ending with endAllocate ##";
			mesh->endAllocate();
			if (is_parallel){
			  info() << "[XmfMeshReader] ## setOwnersFromCells ##";
			  mesh->setOwnersFromCells();
			}
			info() << "\n\n[XmfMeshReader] ## Now dealing with ghost's layer ##";
			info() << "[XmfMeshReader] mesh.nbNode=" <<mesh->nbNode() << " mesh.nbCell="<< mesh->nbCell();

			/***********************************************************
			 and zero or more Attribute elements (fetching Other Groups)
			************************************************************/
			XdmfInt32 nAttributes=Grid->GetNumberOfAttributes();
			info() << "nAttributes=" << nAttributes;
			for(XdmfInt64 iAttribute=0; iAttribute < nAttributes; ++iAttribute){
				XdmfAttribute *attribute = Grid->GetAttribute(iAttribute);
				if ((attribute == NULL || (!itWasAnArcanProduction))) continue;
				if ((strcasecmp(attribute->GetName(), "NodesUniqueIDs")==0)||
					 (strcasecmp(attribute->GetName(), "CellsUniqueIDs")==0)||
					 (strcasecmp(attribute->GetName(), "NodesOwner")==0)){
				  info() << "Skipping " << attribute->GetName();
				  continue;
				}
				attribute->Update();
				XdmfArray *xmfGroup = attribute->GetValues();
				info() << attribute->GetName() << "(" << attribute->GetAttributeTypeAsString() << ", "
						 << attribute->GetAttributeCenterAsString() << ", " << xmfGroup->GetNumberOfElements() <<")";

				eItemKind itemKind=(eItemKind)xmfGroup->GetValueAsInt32(0);
				IItemFamily* family = mesh->itemFamily(itemKind);
				Integer nb_item = xmfGroup->GetNumberOfElements() - 1;
				Int64UniqueArray unique_ids(nb_item);

				// Les éléments suivant contiennent les uniqueId() des entités du groupe.
				for(XdmfInt64 z=0; z<nb_item; ++z )
				  unique_ids[z] = xmfGroup->GetValueAsInt32(z+1);

				// Récupère le localId() correspondant.
				Int32UniqueArray local_ids(unique_ids.size());
				family->itemsUniqueIdToLocalId(local_ids,unique_ids,false);
				
				// Tous les entités ne sont pas forcément dans le maillage actuel et
				// il faut donc les filtrer.
				Int32UniqueArray ids;
				for(Integer i=0; i<nb_item; ++i )
				  if (local_ids[i]!=NULL_ITEM_LOCAL_ID)
					 ids.add(local_ids[i]);
				
				info() << "Create group family=" << family->name() << " name=" << attribute->GetName() << " ids=" << ids.size();
				family->createGroup(attribute->GetName(),ids,true);
			}

			/*********************
			 * Now insert coords *
			 *********************/
			info() << "[XmfMeshReader] ##  Now insert coords ##";
			VariableNodeReal3& nodes_coord_var(mesh->nodesCoordinates());
			ENUMERATE_NODE(iNode,mesh->ownNodes()){
			  nodes_coord_var[iNode] = nodes_coords[iNode->uniqueId()];
			}
			
			/****************************************
			 * Synchronizing groups/variables & nodes *
			 ****************************************/
			mesh->synchronizeGroupsAndVariables();
		}
	}

	info() << "[readMeshFromFile] RTOk";
	//	delete DOM;
	// delete geometry;
	// delete topology;
	return RTOk;
}
  
  

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
