// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* XmfMeshWriter.cc                                            (C) 2000-2025 */
/*                                                                           */
/* Ecriture d'un fichier au format Xmf.                                      */
/*****************************************************************************
/* TODO: - Work on Precision (="4"), which could be adjusted.
         - Test for new output file
*/
/*****************************************************************************
 * [Topology]		 																				
 *****************************************************************************
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
******************************************************************************
XdmfTopology has the general class (XDMF_STRUCTURED | XDMF_UNSTRUCTURED)
and the specific BASE type (TETRAHEDRON | 3DSMESH etc.).
******************************************************************************
For unstructured meshes, XdmfTopology also contains the connectivity array.
For structured meshes, connectivity is implicit (i.e. X[i] is connected to X[i+1])
*****************************************************************************/

/*****************************************************************************
 * [Geometry]				 																		
 *****************************************************************************
	XdmfGeometry is a required part of an XdmfGrid. Geometry can be specified in several different ways :

	XDMF_GEOMETRY_XYZ 	: X0,Y0,Z0,X1,Y1,Z1 ..... XN,YN,ZN  for every point
	XDMF_GEOMETRY_X_Y_Z	: X0,X1 ... XN,Y0,Y1 ... YN,Z0,Z1 ... ZN  for every point
	XDMF_GEOMETRY_VXVYVZ	: X0,X1 ... XN,Y0,Y1 ... YN,Z0,Z1 ... ZN for XAxis, YAxis, ZAxis
	XDMF_GEOMETRY_ORIGIN_DXDYDZ : Xorigin, Yorigin, Zorigin, Dx, Dy, Dz
	******************************************************************************
    XML Element : Grid
    XML Attribute : Name = Any String
    XML Attribute : GeometryType = XYZ* | XY | X_Y_Z | X_Y | VXVYVZ | ORIGIN_DXDYDZ
	 ******************************************************************************
    Example :
        <Grid Name="Mesh" GridType="Uniform">
            <Topology ...
            <Geometry ...
            <Attribute ...
        </Grid>
*****************************************************************************/

/***************************************************************************** \
* [DataItem]				 																		*
******************************************************************************
An XdmfDataItem is a container for data. It is of one of these types :

    Uniform ...... A single DataStructure
    HyperSlab .... A DataTransform that Subsamples some DataStructure
    Coordinates .. A DataTransform that Subsamples via Parametric Coordinates
    Function ..... A DataTransform described by some function
    Collection ... Contains an Array of 1 or more DataStructures or DataTransforms
    Tree ......... A Hierarchical group of other DataItems

If not specified in the "ItemType" a Uniform item is assumed.
A Uniform DataItem is a XdmfDataStructure or an XdmfDataTransform.
Both XdmfDataStructure and XdmfDataTransform are maintined for backwards compatibility.
******************************************************************************
 XML Attribute : Name = Any String, DataItems have an optional name.
 XML Attribute : ItemType = Uniform* | Collection | Tree | HyperSlab | Coordinates | Function
 XML Attribute : Dimensions = K J I 
		Dimensions are listed with the slowest varying dimension first.
		(i.e. KDim JDim IDim).
 XML Attribute : NumberType = Float* | Int | UInt | Char | UChar
		Type is "Char | Float | Int | Compound" with the default being Float.
 XML Attribute : Precision = 1 | 4 | 8
		Precision is BytesPerElement and defaults to 4 for Ints and Floats.
 XML Attribute : Format = XML* | HDF
		Format is any supported XDMF format but usually XML | HDF. 
\*****************************************************************************/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Iostream.h"
#include "arcane/utils/StdHeader.h"
#include "arcane/utils/HashTableMap.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/String.h"
#include "arcane/utils/IOException.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/Collection.h"
#include "arcane/utils/Enumerator.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/Real3.h"

#include "arcane/core/FactoryService.h"
#include "arcane/core/IMainFactory.h"
#include "arcane/core/IMeshReader.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/IMesh.h"
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

#define XDMF_USE_ANSI_STDLIB
#include <vtkxdmf2/Xdmf.h>


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
namespace Arcane
{

using namespace xdmf2;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Ecriture des fichiers de maillage aux format xmf.
 */
class XmfMeshWriter
: public AbstractService
, public IMeshWriter
{
 public:
	explicit XmfMeshWriter(const ServiceBuildInfo& sbi)
  : AbstractService(sbi)
  {
		this->CurrIndent=0;
	}

  virtual void build(void) {}
	virtual bool writeMeshToFile(IMesh* mesh,const String& file_name);

	bool xmfWriteHead(void);
	bool xmfWriteTail(void);

	bool xmfWriteDomainHeader(void);
	bool xmfWriteDomainFooter(void);
	
	bool xmfWriteGridHeader(char*, char*);
	bool xmfWriteGridFooter(void);
	
	bool xmfWriteTopologyHeader(char*, XdmfInt32);
	bool xmfWriteTopologyFooter(void);

	bool xmfWriteGeometryHeader(char *Name, XdmfInt32 GeometryType);
	bool xmfWriteGeometryFooter(void);

	bool xmfWriteDataItemHeader(XdmfInt32[3], XdmfInt32, XdmfInt32);
	bool xmfWriteDataItemFooter(void);

	void Indent(void);
	void IncrementIndent() { this->CurrIndent ++; }
	void DecrementIndent() { if (this->CurrIndent >= 0) this->CurrIndent--; }

private:
  ofstream ost;
  void _switchXmfType(Integer, Array<Integer>&);
  int CurrIndent;
  char *HeavyDataSetNameString;
};


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
ARCANE_REGISTER_SUB_DOMAIN_FACTORY(XmfMeshWriter,IMeshWriter,XmfNewMeshWriter);
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*****************************************************************************\
* [_switchXmfType]			 																	*
\*****************************************************************************/
void XmfMeshWriter::
_switchXmfType(Integer arc_type, Array<Integer>& arcConnectivityArray)
{
  if (arc_type > ItemTypeMng::nbBasicItemType()){
	 arcConnectivityArray.add(XDMF_NOTOPOLOGY);
	 return;
  }
  
  switch(arc_type){
  case (IT_NullType): arcConnectivityArray.add(XDMF_NOTOPOLOGY);return;
	 
  case (IT_Vertex):
	 arcConnectivityArray.add(XDMF_POLYVERTEX);
	 arcConnectivityArray.add(1ul); return;
	 
  case (IT_Line2):
	 arcConnectivityArray.add(XDMF_POLYLINE);
	 arcConnectivityArray.add(2ul); return;
	 
  case (IT_Triangle3): arcConnectivityArray.add(XDMF_TRI); return;
  case (IT_Quad4): arcConnectivityArray.add(XDMF_QUAD);  return;
  case (IT_Pentagon5): arcConnectivityArray.add(XDMF_POLYGON); return;
  case (IT_Hexagon6): arcConnectivityArray.add(XDMF_POLYGON); return;
  case (IT_Heptagon7): arcConnectivityArray.add(XDMF_POLYGON); return;
  case (IT_Octogon8): arcConnectivityArray.add(XDMF_POLYGON); return;
  case (IT_Tetraedron4): arcConnectivityArray.add(XDMF_TET); return;
  case (IT_Pyramid5): arcConnectivityArray.add(XDMF_PYRAMID); return;
  case (IT_Pentaedron6): arcConnectivityArray.add(XDMF_WEDGE); return;
  case (IT_Hexaedron8):	arcConnectivityArray.add(XDMF_HEX); return;
  case (IT_Heptaedron10):	arcConnectivityArray.add(XDMF_TET_10); return;
	 
  case (IT_Octaedron12):
	 arcConnectivityArray.add(XDMF_POLYVERTEX);
	 arcConnectivityArray.add(12ul); return;
case (IT_Enneedron14):
         arcConnectivityArray.add(XDMF_POLYVERTEX);
         arcConnectivityArray.add(14ul); return;
 case (IT_Decaedron16):
         arcConnectivityArray.add(XDMF_POLYVERTEX);
         arcConnectivityArray.add(16ul); return;
  case (IT_HemiHexa7):
	 arcConnectivityArray.add(XDMF_POLYVERTEX);
	 arcConnectivityArray.add(7ul); return;

	 //warning IT_AntiWedgeLeft6 IT_AntiWedgeRight6 and IT_HemiHexa6 are merged
  case (IT_HemiHexa6):
  case (IT_AntiWedgeLeft6):
  case (IT_AntiWedgeRight6):
	 arcConnectivityArray.add(XDMF_POLYVERTEX);
	 arcConnectivityArray.add(6ul); return;
	 
	 //warning IT_HemiHexa5 and IT_DiTetra5 are merged
  case (IT_HemiHexa5):
  case (IT_DiTetra5):	
	 arcConnectivityArray.add(XDMF_POLYVERTEX);
	 arcConnectivityArray.add(5ul); return;
	 
  case (IT_DualNode):
  case (IT_DualEdge):
  case (IT_DualFace):
  case (IT_DualCell): arcConnectivityArray.add(XDMF_NOTOPOLOGY); return;
	 
  default: arcConnectivityArray.add(XDMF_NOTOPOLOGY);return;
  }
  arcConnectivityArray.add(XDMF_NOTOPOLOGY);
}




/**********************************************************************
 * [writeMeshToFile]																	  
 **********************************************************************/
bool XmfMeshWriter::
writeMeshToFile(IMesh* mesh,const String& file_name)
{
  info() << "[XmfMeshWriter::writeMeshToFile] nNodes=" <<mesh->nbNode() << " nCells="<< mesh->nbCell()
			 << " all=" << mesh->allNodes().size() << ", own=" << mesh->ownNodes().size();

  /****************************
	* XDMF-side initialisation *
	****************************/
	XdmfRoot* xmfRoot=new XdmfRoot(); // represents the Root Element in Xdmf
	XdmfDOM* xmfDom= new XdmfDOM();
  String h5_file_name = file_name + ".h5";
	if (platform::isFileReadable(h5_file_name))
	  if (platform::removeFile(h5_file_name))
			ARCANE_FATAL("Could not remove .h5 file '{0}'",h5_file_name);
	String xmfDomFileName(file_name);
  // Add extension '.xmf' if needed.
  if (!xmfDomFileName.endsWith(".xmf"))
    xmfDomFileName = file_name + ".xmf";
	if (xmfDom->SetWorkingDirectory(".")!= XDMF_SUCCESS)
    throw IOException("writeMeshToFile", "SetOutputFileName");
	if (xmfDom->SetOutputFileName(xmfDomFileName.localstr())!= XDMF_SUCCESS)
    throw IOException("writeMeshToFile", "SetOutputFileName");
	xmfRoot->SetDOM(xmfDom);
	xmfRoot->Build();
	info() << "XDMF-side initialisation done filename='" << xmfDomFileName << "'";

	// Domain initialisation
	XdmfDomain *xmfDomain = new XdmfDomain();
	xmfDomain->SetName(file_name.localstr());
	xmfRoot->Insert(xmfDomain);
	info() << "[XmfMeshWriter] Domain initialisation done";

	// Grid initialisation
	XdmfGrid *xmfGrid = new XdmfGrid();
	xmfGrid->SetGridType(XDMF_GRID_UNIFORM);
	info() << "[XmfMeshWriter] Grid initialisation done";

	/****************************************
	 * XdmfAttribute to save Cells Unique IDs 
	 ****************************************/
	XdmfAttribute *xmfCellAttribute=new XdmfAttribute();
	xmfCellAttribute->SetName("CellsUniqueIDs");
	xmfCellAttribute->SetAttributeCenter(XDMF_ATTRIBUTE_CENTER_CELL);
	xmfCellAttribute->SetAttributeType(XDMF_ATTRIBUTE_TYPE_SCALAR);
	XdmfArray *xmfCellsUniqueIDs = xmfCellAttribute->GetValues();
	String heavyDataForCellsUniqueIDs(h5_file_name+":/CellsUniqueIDs");
	xmfCellsUniqueIDs->SetHeavyDataSetName(heavyDataForCellsUniqueIDs.localstr());
	xmfCellsUniqueIDs->SetNumberType(XDMF_INT32_TYPE);
	xmfCellsUniqueIDs->SetNumberOfElements(mesh->nbCell());
	XdmfInt64 cellIndex=0;

	/****************************************
	 * XdmfAttribute to save Nodes Unique IDs
	 ****************************************/
	XdmfAttribute *xmfNodeAttribute=new XdmfAttribute();
	xmfNodeAttribute->SetName("NodesUniqueIDs");
	xmfNodeAttribute->SetAttributeCenter(XDMF_ATTRIBUTE_CENTER_NODE);
	xmfNodeAttribute->SetAttributeType(XDMF_ATTRIBUTE_TYPE_SCALAR);
	XdmfArray *xmfNodesUniqueIDs = xmfNodeAttribute->GetValues();
	String heavyDataForNodesUniqueIDs(h5_file_name+":/NodesUniqueIDs");
	xmfNodesUniqueIDs->SetHeavyDataSetName(heavyDataForNodesUniqueIDs.localstr());
	xmfNodesUniqueIDs->SetNumberType(XDMF_INT32_TYPE);
	xmfNodesUniqueIDs->SetNumberOfElements(mesh->nbNode());
	IntegerUniqueArray nodesUniqueIDs; // Unique nodes-IDs array
	
	/***********************************
	* XdmfAttribute to save Nodes Owners
	\***********************************/
	XdmfAttribute *xmfOwnerAttribute=new XdmfAttribute();
	xmfOwnerAttribute->SetName("NodesOwner");
	xmfOwnerAttribute->SetAttributeCenter(XDMF_ATTRIBUTE_CENTER_NODE);
	xmfOwnerAttribute->SetAttributeType(XDMF_ATTRIBUTE_TYPE_SCALAR);
	XdmfArray *xmfNodesOwners = xmfOwnerAttribute->GetValues();
	String heavyDataForOwners(h5_file_name+":/NodesOwners");
	xmfNodesOwners->SetHeavyDataSetName(heavyDataForOwners.localstr());
	xmfNodesOwners->SetNumberType(XDMF_INT32_TYPE);
	xmfNodesOwners->SetNumberOfElements(mesh->nbNode());
	XdmfInt64 nodeIndex=0;
	ENUMERATE_NODE(iNode,mesh->allNodes()){
	  nodesUniqueIDs.add(iNode->uniqueId().asInteger());
	  xmfNodesUniqueIDs->SetValue(nodeIndex, iNode->uniqueId().asInteger());
	  xmfNodesOwners->SetValue(nodeIndex++, iNode->owner());
	}
	
	// Each Grid contains a Topology,
	info() << "[XmfMeshWriter] Focussing on the topology";
	XdmfTopology *xmfTopology = xmfGrid->GetTopology();
	xmfTopology->SetTopologyType(XDMF_MIXED);
	xmfTopology->SetNumberOfElements(mesh->nbCell());
	XdmfArray *xmfConnectivityArray= xmfTopology->GetConnectivity();
	String heavyDataForConnections(h5_file_name+":/Connections");
	xmfConnectivityArray->SetHeavyDataSetName(heavyDataForConnections.localstr());
	xmfConnectivityArray->SetNumberType(XDMF_INT32_TYPE);
	UniqueArray<XdmfInt32> arcCellConnectivityArray;
	ENUMERATE_CELL(iCell,mesh->allCells()){// Scanning the cells' nodes to get type and connectivity
	  Cell cell = *iCell;
	  Integer  nbNodes = cell.nbNode();
	  xmfCellsUniqueIDs->SetValue(cellIndex++, iCell->uniqueId().asInteger());
	  _switchXmfType(cell.type(), arcCellConnectivityArray);
	  Integer meshNbNodes=mesh->nbNode();
	  for( Integer j=0; j<nbNodes;++j){
      Integer uid=cell.node(j).uniqueId();
      for( Integer i=0; i<meshNbNodes; ++i){
        // This is used for external viewers to be able to read our output
        if (nodesUniqueIDs[i] != uid)
          continue; // xmfNodesUniqueIDs->GetValueAsInt32(i) is just TOO painful to work with!
        arcCellConnectivityArray.add(i);
        break;
      }
	  }
	}
	xmfConnectivityArray->SetNumberOfElements(arcCellConnectivityArray.size());
	info() << "[XmfMeshWriter] arcCellConnectivityArray.size()=" << arcCellConnectivityArray.size();
	for(XdmfInt32 idx=0; idx<arcCellConnectivityArray.size();++idx)
	  xmfConnectivityArray->SetValue(idx,arcCellConnectivityArray[idx]);
	info() << "[XmfMeshWriter] Work on grid->topology done";

	// a Geometry,
	XdmfGeometry *xmfGeometry = xmfGrid->GetGeometry();
	xmfGeometry->SetGeometryType(XDMF_GEOMETRY_XYZ);
	//xmfGeometry->SetNumberOfPoints(mesh->nbNode());  
	XdmfArray *xmfNodeGeometryArray= xmfGeometry->GetPoints();
	String heavyDataForGeometry(h5_file_name+":/XYZ");
	xmfNodeGeometryArray->SetHeavyDataSetName(heavyDataForGeometry.localstr());
	xmfNodeGeometryArray->SetNumberType(XDMF_FLOAT32_TYPE);
	xmfNodeGeometryArray->SetNumberOfElements(3*mesh->nbNode());// Number of points in this geometry
	VariableItemReal3& nodes_coords = mesh->nodesCoordinates();
	XdmfInt64 Index=0;
	ENUMERATE_NODE(iNode,mesh->allNodes()){
	  const Node& node = *iNode;
	  xmfNodeGeometryArray->SetValue(Index++,Convert::toDouble(nodes_coords[iNode].x));
	  xmfNodeGeometryArray->SetValue(Index++,Convert::toDouble(nodes_coords[iNode].y));
	  xmfNodeGeometryArray->SetValue(Index++,Convert::toDouble(nodes_coords[iNode].z));
	  //info() << "[writeMeshToFile] Adding node[" << iNode->uniqueId() << "]";
	}
	info() << "[XmfMeshWriter] Work on Geometry done";
	xmfDomain->Insert(xmfGrid); 

	
	/*************************
	 * Fetching Other Groups 
	 * XML Attribute : Name
	 *	XML Attribute : AttributeType = Scalar* | Vector | Tensor | Tensor6 | Matrix
	 * XML Attribute : Center = Node* | Cell | Grid | Face | Edge
	 ************************/
	info() << "[XmfMeshWriter] Working on Groups";
	for(ItemGroupCollection::Enumerator arcGroup(mesh->groups()); ++arcGroup;){
      if (	(*arcGroup == mesh->cellFamily()->allItems())||(*arcGroup == mesh->nodeFamily()->allItems())||\
				(*arcGroup == mesh->edgeFamily()->allItems())||(*arcGroup == mesh->faceFamily()->allItems())) continue;
		info() << "[writeMeshToFile] Found a " << arcGroup->itemKind() << "-group "  << arcGroup->name();
		XdmfAttribute *xmfAttribute=new XdmfAttribute();
		xmfAttribute->SetName(arcGroup->name().localstr());
		xmfAttribute->SetAttributeCenter(XDMF_ATTRIBUTE_CENTER_NODE);
		xmfAttribute->SetAttributeType(XDMF_ATTRIBUTE_TYPE_SCALAR);
		XdmfArray *xmfGroup = xmfAttribute->GetValues();
		String heavyDataForGroup(h5_file_name+":/" + arcGroup->name());
		xmfGroup->SetHeavyDataSetName(heavyDataForGroup.localstr());
		xmfGroup->SetNumberType(XDMF_INT32_TYPE);
		xmfGroup->SetNumberOfElements(1+arcGroup->size());
		XdmfInt64 Index=0;
		xmfGroup->SetValue(Index++, arcGroup->itemKind());
		ENUMERATE_ITEM(iItem, *arcGroup){
		  xmfGroup->SetValue(Index++, iItem->uniqueId().asInteger());
		}
		xmfGrid->Insert(xmfAttribute);
	}
	xmfGrid->Insert(xmfCellAttribute);
	xmfGrid->Insert(xmfNodeAttribute);
	xmfGrid->Insert(xmfOwnerAttribute);
	xmfGrid->Build();

	
	/********************
	 * Output & cleanup *
	 ********************/
	xmfDom->Write();

	delete xmfCellAttribute;
	delete xmfNodeAttribute;
	delete xmfOwnerAttribute;
	delete xmfGrid;
	delete xmfDomain;
	delete xmfRoot;
	delete xmfDom;
	
	info() << "[XmfMeshWriter] Done";
	return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
