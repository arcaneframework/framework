// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MshMeshWriter.cc                                            (C) 2000-2021 */
/*                                                                           */
/* Lecture/Ecriture d'un fichier au format MSH.                              */
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
#include "arcane/utils/Collection.h"
#include "arcane/utils/Enumerator.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/Real3.h"

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
#include "arcane/SharedVariable.h"

#include "arcane/AbstractService.h"

/*****************************************************************************\
* DEFINES						 																	*
* Element types in .msh file format, found in gmsh-2.0.4/Common/GmshDefines.h *
\*****************************************************************************/

#include "arcane/std/internal/IosGmsh.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ecriture des fichiers de maillage aux format msh.
 */
class MshMeshWriter
: public AbstractService
, public IMeshWriter
{
 public:

	MshMeshWriter(const ServiceBuildInfo& sbi) : AbstractService(sbi){}
	virtual void build() {}
	virtual bool writeMeshToFile(IMesh* mesh,const String& file_name);

 private:

	Integer _switchMshType(Integer mshElemType);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SUB_DOMAIN_FACTORY(MshMeshWriter,IMeshWriter,MshNewMeshWriter);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*****************************************************************************\
* [_switchMshType]			 																	*
\*****************************************************************************/
/*!/brief Selon le type MSH passé en argument, cette fonction retourne l'Integer
correspondant  type d'ARCANE. Une exception est levée dans le cas d'une inadéquation.
\param msh_type Type MSH proposé au décodage
\retrun Le type ARCANE trouvé en correspondance
*/
Integer MshMeshWriter::_switchMshType(Integer msh_type){
	switch (msh_type){
//		case (IT_NullType):			return MSH_LIN_2;		//case (0) is not used
		case (IT_Vertex):	  			return MSH_PNT;  		//printf("1-node point");  					
		case (IT_Line2): 				return MSH_LIN_2;		//printf("2-node line");						
		case (IT_Triangle3): 		return MSH_TRI_3;  	//printf("3-node triangle");  				
		case (IT_Quad4): 				return MSH_QUA_4;		//printf("4-node quadrangle");				
		case (IT_Tetraedron4):		return MSH_TET_4;		//printf("4-node tetrahedron");  			
		case (IT_Hexaedron8): 		return MSH_HEX_8; 	//printf("8-node hexahedron");				
		case (IT_Pentaedron6):		return MSH_PRI_6;		//printf("6-node prism");  					
		case (IT_Pyramid5):			return MSH_PYR_5;		//printf("5-node pyramid");
		// Beneath, are some meshes that have been tried to match gmsh's ones
		// Other 5-nodes
		case (IT_Pentagon5):			return MSH_PYR_5;		// Could use a tag to encode these
		case (IT_HemiHexa5):			return MSH_PYR_5;	
		case (IT_DiTetra5):			return MSH_PYR_5;
		// Other 6-nodes
		case (IT_Hexagon6):			return MSH_PRI_6;
		case (IT_HemiHexa6):			return MSH_PRI_6;	
		case (IT_AntiWedgeLeft6):	return MSH_PRI_6;
		case (IT_AntiWedgeRight6):	return MSH_PRI_6;
		// Other 10-nodes
		case (IT_Heptaedron10):		return MSH_TRI_10;
		// Other 12-nodes
		case (IT_Octaedron12):		return MSH_TRI_12; 
		// Other ?-nodes, have to work with another field (nNodes)
		case (IT_HemiHexa7):			return IT_NullType;	// This IT_NullType will be associated with its number of nodes
		// Others ar still considered as default, rising an exception
		case (IT_DualNode):
		case (IT_DualEdge):
		case (IT_DualFace):
		case (IT_DualCell):
		default:
			info() << "_switchMshType Non supporté (" << msh_type << ")";
			throw IOException("_switchMshType Non supporté");
	}// Not found: IT_Pentagon5, IT_Hexagon6, IT_Octaedron12
	info() << "_switchMshType non switché (" << msh_type << ")";
	throw IOException("_switchMshType non switché");
	return 0;
}



/**********************************************************************\
* [writeMeshToFile]																	  *
\**********************************************************************/
/*!\brief writeMeshToFile écrit au format gmsh tel que spécifié ci-dessous:
\code
	$MeshFormat
	2.0 file-type data-size
	$EndMeshFormat
	$Nodes
	number-of-nodes
	node-number x-coord y-coord z-coord
	...
	$EndNodes
	$Elements
	number-of-elements
	elm-number elm-type number-of-tags < tag > ... node-number-list
	...
	$EndElements
\endcode
	
	\param mesh Maillage d'entrée
	\param file_name Nom du fichier de sortie
	\return
		- true	Pour toute erreur détectée
		- false	Sinon
*/
bool MshMeshWriter::
writeMeshToFile(IMesh* mesh,const String& file_name)
{
  String mshFileName(file_name+".msh");
  std::ofstream ofile(mshFileName.localstr());
	if (!ofile)
		throw IOException("VtkMeshIOService::writeMeshToFile(): Unable to open file");

	info() << "[writNodes=" << mesh->nbNode() << " nCells="<< mesh->nbCell();

	ofile << "$MeshFormat\n";
	ofile << "2.0 0 " << (int) sizeof(double) << "\n";
	ofile << "$EndMeshFormat\n";
	ofile << "$Nodes\n";
	ofile << mesh->nbNode() << "\n";
	
	SharedVariableNodeReal3 nodes_coords = mesh->sharedNodesCoordinates();
	ItemGroup all_nodes = mesh->allNodes();

	ENUMERATE_NODE(inode,all_nodes){
    Node node = *inode;
		Real3 coord = nodes_coords[inode];
		double vtkXyz[3];

    vtkXyz[0] = Convert::toDouble(coord.x);
    vtkXyz[1] = Convert::toDouble(coord.y);
    vtkXyz[2] = Convert::toDouble(coord.z);
    ofile << node.uniqueId() << " " << vtkXyz[0] << " " << vtkXyz[1] << " " <<vtkXyz[2] << "\n";
		//info() << "[writeMeshToFile] Adding node[" << node.uniqueId().asInt64() << "]";
	}
  
	ofile << "$EndNodes\n";
	ofile << "$Elements\n";
	ofile << mesh->nbCell() << "\n";
	
  // Scanning the cells' nodes to get type and connectivity
	// elm-number elm-type number-of-tags < tag > ... node-number-list
	ENUMERATE_CELL(iCell,mesh->allCells()){
		Cell cell = *iCell;
		int  nbNodes = cell.nbNode();
		ofile << cell.uniqueId().asInt64() << " " << _switchMshType(cell.type());
		if (_switchMshType(cell.type()) ==  IT_NullType) 
			ofile  << " 1 " << nbNodes << " ";
		else
			ofile  << " 0 ";
		for(Integer j=0; j<nbNodes;++j)
			ofile << cell.node(j).uniqueId().asInt64() << " ";
		ofile << "\n";
	}

	ofile << "$EndElements\n";
	return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
