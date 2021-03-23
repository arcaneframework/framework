// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MshMeshReader.cc                                            (C) 2000-2021 */
/*                                                                           */
/* Lecture/Ecriture d'un fichier au format MSH.				                       */
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
#include "arcane/MeshPartInfo.h"
#include "arcane/ios/IosFile.h"

// Element types in .msh file format, found in gmsh-2.0.4/Common/GmshDefines.h
#include "arcane/ios/IosGmsh.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * NOTES:
 * - La bibliothèque `gmsh` fournit un script 'open.py' dans le répertoire
 *   'demos/api' qui permet de générer un fichier '.msh' à partir d'un '.geo'.
 *
 * TODO:
 * - supporter les partitions
 * - supporter les groupes pour les conditions aux limites
 * - pouvoir utiliser la bibliothèque 'gmsh' directement.
 * - supporter ce format avec les services de maillage.
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Lecteur de fichiers de maillage au format `msh`.
 *
 * Le format `msh` est celui utilisé par la bibliothèque 
 * [gmsh](https://gmsh.info/).
 *
 * Le lecteur supporte les versions `2.0` et `4.1` de ce format.
 *
 * Seules une partie des fonctionnalités du format sont supportées:
 *
 * - seuls les éléments d'ordre 1 sont supportés.
 * - les coordonnées paramétriques ne sont pas supportées
 * - seules les sections `$Nodes` et `$Entities` sont lues
 */
class MshMeshReader
: public AbstractService
, public IMeshReader
{
 public:

  /*!
   * \brief Infos d'un bloc pour $Elements pour la version 4.
   *
   * Dans cette version, les éléments d'un bloc sont tous
   * de même type (par exemple que des IT_Quad4 ou IT_Triangle3.
   */
  struct MeshV4ElementsBlock
  {
    Integer index = 0; //!< Index du bloc dans la liste
    Integer nb_entity = 0; //!< Nombre d'entités du bloc
    Integer cell_type = -1; //!< Type Arcane de l'entité
    Integer dimension = -1; //!< Dimension de l'entité
    Integer cell_nb_node = 0; //!< Nombre de noeuds de l'entité.
    UniqueArray<Int64> connectivity;
  };

  struct MeshInfo
  {
    MeshInfo() : node_coords_map(5000,true){}
    Integer nb_elements = 0;
    Integer nb_cell_node = 0;
    UniqueArray<Integer> cells_nb_node;
    UniqueArray<Int64> cells_connectivity;
    UniqueArray<Integer> cells_type;
    UniqueArray<Real3> node_coords;
    HashTableMapT<Int64, Real3> node_coords_map;
  };

 public:

  explicit MshMeshReader(const ServiceBuildInfo& sbi);

 public:
  void build() override {}

  bool allowExtension(const String& str) override { return str == "msh"; }

  eReturnType readMeshFromFile(IPrimaryMesh* mesh, const XmlNode& mesh_node,
                               const String& file_name, const String& dir_name,
                               bool use_internal_partition) override
  {
    ARCANE_UNUSED(dir_name);
    return _readMeshFromMshFile(mesh, mesh_node, file_name, use_internal_partition);
  }


 private:

  Integer m_version = 0;

  eReturnType _readNodesFromAsciiMshV2File(IosFile&, Array<Real3>&);
  eReturnType _readNodesFromAsciiMshV4File(IosFile&, MeshInfo& mesh_info);
  eReturnType _readNodesFromBinaryMshFile(IosFile&, Array<Real3>&);
  Integer _readElementsFromAsciiMshV2File(IosFile&, MeshInfo& mesh_info);
  Integer _readElementsFromAsciiMshV4File(IosFile&, MeshInfo& mesh_info);
  eReturnType _readMeshFromNewMshFile(IMesh*, IosFile&);
  eReturnType _createMeshFrom(IMesh* mesh, MeshInfo& mesh_info);
  Integer _switchMshType(Integer, Integer&);
  eReturnType _readMeshFromMshFile(IMesh* mesh, const XmlNode& mesh_node,
                                   const String& file_name, bool use_internal_partition);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(MshMeshReader,
                        ServiceProperty("MshNewMeshReader",ST_SubDomain),
                        ARCANE_SERVICE_INTERFACE(IMeshReader));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MshMeshReader::
MshMeshReader(const ServiceBuildInfo& sbi)
: AbstractService(sbi)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer MshMeshReader::
_switchMshType(Integer mshElemType, Integer& nNodes)
{
  switch (mshElemType) {
  case (IT_NullType): // used to decode IT_NullType: IT_HemiHexa7|IT_Line9
    switch (nNodes) {
    case (7):
      return IT_HemiHexa7;
    default:
      info() << "Could not decode IT_NullType with nNodes=" << nNodes;
      throw IOException("_switchMshType", "Could not decode IT_NullType with nNodes");
    }
    break;
  case (MSH_PNT):
    nNodes = 1;
    return IT_Vertex; //printf("1-node point");
  case (MSH_LIN_2):
    nNodes = 2;
    return IT_Line2; //printf("2-node line");
  case (MSH_TRI_3):
    nNodes = 3;
    return IT_Triangle3; //printf("3-node triangle");
  case (MSH_QUA_4):
    nNodes = 4;
    return IT_Quad4; //printf("4-node quadrangle");
  case (MSH_TET_4):
    nNodes = 4;
    return IT_Tetraedron4; //printf("4-node tetrahedron");
  case (MSH_HEX_8):
    nNodes = 8;
    return IT_Hexaedron8; //printf("8-node hexahedron");
  case (MSH_PRI_6):
    nNodes = 6;
    return IT_Pentaedron6; //printf("6-node prism");
  case (MSH_PYR_5):
    nNodes = 5;
    return IT_Pyramid5; //printf("5-node pyramid");
  case (MSH_TRI_10):
    nNodes = 10;
    return IT_Heptaedron10; //printf("10-node second order tetrahedron")
  case (MSH_TRI_12):
    nNodes = 12;
    return IT_Octaedron12; //printf("Unknown MSH_TRI_12");

  case (MSH_TRI_6):
  case (MSH_QUA_9):
  case (MSH_HEX_27):
  case (MSH_PRI_18):
  case (MSH_PYR_14):
  case (MSH_QUA_8):
  case (MSH_HEX_20):
  case (MSH_PRI_15):
  case (MSH_PYR_13):
  case (MSH_TRI_9):
  case (MSH_TRI_15):
  case (MSH_TRI_15I):
  case (MSH_TRI_21):
  default:
    ARCANE_THROW(NotSupportedException, "Unknown GMSH element type '{0}'", mshElemType);
  }
  return IT_NullType;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshReader::eReturnType MshMeshReader::
_readNodesFromAsciiMshV2File(IosFile& ios_file, Array<Real3>& node_coords)
{
  // number-of-nodes & coords
  info() << "[_readNodesFromAsciiMshV2File] Looking for number-of-nodes";
  Integer nb_node = ios_file.getInteger();
  if (nb_node < 0)
    throw IOException(A_FUNCINFO, String("Invalid number of nodes: n=") + nb_node);
  info() << "[_readNodesFromAsciiMshV2File] nb_node=" << nb_node;
  for (Integer i = 0; i < nb_node; ++i) {
    // Il faut lire l'id même si on ne s'en sert pas.
    Integer id = ios_file.getInteger();
    ARCANE_UNUSED(id);
    Real nx = ios_file.getReal();
    Real ny = ios_file.getReal();
    Real nz = ios_file.getReal();
    node_coords.add(Real3(nx, ny, nz));
    //info() << "id_" << id << " xyz(" << nx << "," << ny << "," << nz << ")";
  }
  ios_file.getNextLine(); // Skip current \n\r
  return RTOk;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 *
 * \code
 $Nodes
 * numEntityBlocks(size_t) numNodes(size_t)
 *   minNodeTag(size_t) maxNodeTag(size_t)
 * entityDim(int) entityTag(int) parametric(int; 0 or 1)
 *   numNodesInBlock(size_t)
 *   nodeTag(size_t)
 *   ...
 *   x(double) y(double) z(double)
 *      < u(double; if parametric and entityDim >= 1) >
 *      < v(double; if parametric and entityDim >= 2) >
 *      < w(double; if parametric and entityDim == 3) >
 *   ...
 * ...
 *$EndNodes
 * \endcode
 */
IMeshReader::eReturnType MshMeshReader::
_readNodesFromAsciiMshV4File(IosFile& ios_file, MeshInfo& mesh_info)
{
  // Première ligne du fichier
  Integer nb_entity = ios_file.getInteger();
  Integer total_nb_node = ios_file.getInteger();
  Integer min_node_tag = ios_file.getInteger();
  Integer max_node_tag = ios_file.getInteger();
  ios_file.getNextLine(); // Skip current \n\r
  if (total_nb_node < 0)
    ARCANE_THROW(IOException,"Invalid number of nodes : '{0}'",total_nb_node);
  info() << "[Nodes] nb_entity=" << nb_entity
         << " total_nb_node=" << total_nb_node
         << " min_tag=" << min_node_tag
         << " max_tag=" << max_node_tag;

  UniqueArray<Int32> nodes_uids;
  for( Integer i_entity=0; i_entity<nb_entity; ++i_entity ){
    // Dimension de l'entité (pas utile)
    [[maybe_unused]] Integer entity_dim = ios_file.getInteger();
    // Tag de l'entité (pas utile)
    [[maybe_unused]] Integer entity_tag = ios_file.getInteger();
    Integer parametric_coordinates = ios_file.getInteger();
    Integer nb_node2 = ios_file.getInteger();
    ios_file.getNextLine();

    info(4) << "[Nodes] entity_dim=" << entity_dim << " entity_tag=" << entity_tag
            << " parametric=" << parametric_coordinates
            << " nb_node2=" << nb_node2;
    if (parametric_coordinates!=0)
      ARCANE_THROW(NotSupportedException,"Only 'parametric coordinates' value of '0' is supported (current={0})",parametric_coordinates);
    // Il est possible que le nombre de noeuds soit 0.
    // Dans ce cas, il faut directement passer à la ligne suivante
    if (nb_node2==0)
      continue;
    nodes_uids.resize(nb_node2);
    for (Integer i = 0; i < nb_node2; ++i) {
      // Conserve le uniqueId() du noeuds.
      nodes_uids[i] = ios_file.getInteger();
      //info() << "I=" << i << " ID=" << id;
    }
    for (Integer i = 0; i < nb_node2; ++i) {
      Real nx = ios_file.getReal();
      Real ny = ios_file.getReal();
      Real nz = ios_file.getReal();
      Real3 xyz(nx, ny, nz);
      mesh_info.node_coords_map.add(nodes_uids[i],xyz);
      //info() << "I=" << i << " COORD=" << xyz;
      //info() << "id_" << id << " xyz(" << nx << "," << ny << "," << nz << ")";
    }
    ios_file.getNextLine(); // Skip current \n\r
  }
  return RTOk;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer MshMeshReader::
_readElementsFromAsciiMshV2File(IosFile& ios_file, MeshInfo& mesh_info)
{
  Integer number_of_elements = ios_file.getInteger(); // is an integer equal to 0 in the ASCII file format, equal to 1 for the binary format
  if (number_of_elements < 0)
    ARCANE_THROW(IOException, String("Error with the number_of_elements=") + number_of_elements);

  info() << "nb_elements=" << number_of_elements;

  //elm-number elm-type number-of-tags < tag > ... node-number-list
  // Lecture des infos des mailles & de la connectivité
  // bool pour voir si on est depuis 0 ou 1
  bool it_starts_at_zero = false;
  for (Integer i = 0; i < number_of_elements; ++i) {
    Integer elm_number = ios_file.getInteger(); // Index
    ARCANE_UNUSED(elm_number);
    Integer elm_type = ios_file.getInteger(); // Type
    // info() << elm_number << " " << elm_type;
    // Now get tags in the case the number of nodes is encoded
    Integer number_of_tags = ios_file.getInteger();
    Integer lastTag = 0;
    for (Integer j = 0; j < number_of_tags; ++j)
      lastTag = ios_file.getInteger();
    Integer number_of_nodes = 0; // Set the number of nodes from the discovered type
    if (elm_type == IT_NullType) {
      number_of_nodes = lastTag;
      info() << "We hit the case the number of nodes is encoded (number_of_nodes=" << number_of_nodes << ")";
    }
    Integer cell_type = _switchMshType(elm_type, number_of_nodes);
    //#warning Skipping 2D lines & points
    // We skip 2-node lines and 1-node points
    if (number_of_nodes < 3) {
      for (Integer j = 0; j < number_of_nodes; ++j)
        ios_file.getInteger();
      continue;
    }

    mesh_info.cells_type.add(cell_type);
    mesh_info.cells_nb_node.add(number_of_nodes);
    info() << elm_number << " " << elm_type << " " << number_of_tags << " number_of_nodes=" << number_of_nodes;
    //		printf("%i %i %i %i %i (", elm_number, elm_type, reg_phys, reg_elem, number_of_nodes);
    for (Integer j = 0; j < number_of_nodes; ++j) {
      //			printf("%i ", node_number);
      Integer id = ios_file.getInteger();
      if (id == 0)
        it_starts_at_zero = true;
      mesh_info.cells_connectivity.add(id);
    }
    //		printf(")\n");
  }
  if (!it_starts_at_zero)
    for (Integer j = 0, max = mesh_info.cells_connectivity.size(); j < max; ++j)
      mesh_info.cells_connectivity[j] = mesh_info.cells_connectivity[j] - 1;

  ios_file.getNextLine(); // Skip current \n\r

  // On ne supporte que les maillage de dimension 3 dans ce vieux format
  return 3;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Lecture des éléments (mailles,faces,...)
 *
 * Dans la version 4, les éléments sont rangés par genre (eItemKind)
 *
 * \code
 *$Elements
 *  numEntityBlocks(size_t) numElements(size_t)
 *    minElementTag(size_t) maxElementTag(size_t)
 *  entityDim(int) entityTag(int) elementType(int; see below)
 *    numElementsInBlock(size_t)
 *    elementTag(size_t) nodeTag(size_t) ...
 *    ...
 *  ...
 *$EndElements
 * \endcode
 */
Integer MshMeshReader::
_readElementsFromAsciiMshV4File(IosFile& ios_file, MeshInfo& mesh_info)
{
  Integer nb_block = ios_file.getInteger();
  
  Integer number_of_elements = ios_file.getInteger(); // is an integer equal to 0 in the ASCII file format, equal to 1 for the binary format
  Integer min_element_tag = ios_file.getInteger();
  Integer max_element_tag = ios_file.getInteger();
  ios_file.getNextLine(); // Skip current \n\r
  info() << "[Elements] nb_block=" << nb_block
         << " nb_elements=" << number_of_elements
         << " min_element_tag=" << min_element_tag
         << " max_element_tag=" << max_element_tag;

  if (number_of_elements < 0)
    ARCANE_THROW(IOException,"Invalid number of elements: {0}",number_of_elements);

  UniqueArray<MeshV4ElementsBlock> blocks(nb_block);
  {
    // Numérote les blocs (pour le débug)
    Integer index = 0;
    for( MeshV4ElementsBlock& block : blocks ){
      block.index = index;
      ++index;
    }
  }
  for( MeshV4ElementsBlock& block : blocks ){
    Integer entity_dim = ios_file.getInteger();
    Integer entity_tag = ios_file.getInteger(); // is an integer equal to 0 in the ASCII file format, equal to 1 for the binary format
    Integer entity_type = ios_file.getInteger();
    Integer nb_entity_in_block = ios_file.getInteger();

    Integer cell_nb_node = 0;
    Integer cell_type = _switchMshType(entity_type, cell_nb_node);

    info(4) << "[Elements] index=" << block.index << " entity_dim=" << entity_dim
            << " entity_tag=" << entity_tag
            << " entity_type=" << entity_type << " nb_in_block=" << nb_entity_in_block
            << " cell_type=" << cell_type << " cell_nb_node=" << cell_nb_node;

    block.nb_entity = nb_entity_in_block;
    block.cell_type = cell_type;
    block.cell_nb_node = cell_nb_node;
    block.dimension = entity_dim;

    for (Integer i = 0; i < nb_entity_in_block; ++i) {
      // TODO: a utiliser
      [[maybe_unused]] Integer cell_unique_id = ios_file.getInteger();

      for ( Integer j = 0; j < cell_nb_node; ++j )
        block.connectivity.add(ios_file.getInteger());
    }
    ios_file.getNextLine(); // Skip current \n\r
  }
  // Maintenant qu'on a tous les blocks, la dimension du maillage est
  // la plus grande dimension des blocks
  Integer mesh_dimension = -1;
  for( MeshV4ElementsBlock& block : blocks )
    mesh_dimension = math::max(mesh_dimension,block.dimension);
  if (mesh_dimension<0)
    ARCANE_FATAL("Invalid computed mesh dimension '{0}'",mesh_dimension);
  if (mesh_dimension!=2 && mesh_dimension!=3)
    ARCANE_THROW(NotSupportedException,"mesh dimension '{0}'. Only 2D or 3D meshes are supported",mesh_dimension);
  info() << "Computed mesh dimension = " << mesh_dimension;

  // On ne conserve que les blocs de notre dimension
  for( MeshV4ElementsBlock& block : blocks ){
    if (block.dimension!=mesh_dimension)
      continue;

    info(4) << "Keeping block index=" << block.index;

    Integer cell_type = block.cell_type;
    Integer cell_nb_node = block.cell_nb_node;

    for (Integer i = 0; i < block.nb_entity; ++i) {
      mesh_info.cells_type.add(cell_type);
      mesh_info.cells_nb_node.add(cell_nb_node);
      auto v = block.connectivity.subView(i*cell_nb_node,cell_nb_node);
      mesh_info.cells_connectivity.addRange(v);
    }
  }

  return mesh_dimension;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * nodes-binary is the list of nodes in binary form, i.e., a array of
 * number-of-nodes *(4+3*data-size) bytes. For each node, the first 4 bytes
 * contain the node number and the next (3*data-size) bytes contain the three
 * floating point coordinates.
 */
IMeshReader::eReturnType MshMeshReader::
_readNodesFromBinaryMshFile(IosFile& ios_file, Array<Real3>& node_coords)
{
  ARCANE_UNUSED(ios_file);
  ARCANE_UNUSED(node_coords);
  return RTError;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshReader::eReturnType MshMeshReader::
_createMeshFrom(IMesh* mesh, MeshInfo& mesh_info)
{
  Integer nb_elements = mesh_info.cells_type.size();
  info() << "nb_of_elements=cells_type.size()=" << nb_elements;
  Integer nb_cell_node = mesh_info.cells_connectivity.size();
  info() << "nb_cell_node=cells_connectivity.size()=" << nb_cell_node;

  IParallelMng* pm = mesh->parallelMng();
  bool is_parallel = pm->isParallel();
  Int32 sid = mesh->meshPartInfo().partRank();
  // Création des mailles
  info() << "Building cells, nb_cell=" << nb_elements << " nb_cell_node=" << nb_cell_node;
  // Infos pour la création des mailles
  // par maille: 1 pour son unique id,
  //             1 pour son type,
  //             1 pour chaque noeud
  UniqueArray<Int64> cells_infos;
  Integer connectivity_index = 0;
  UniqueArray<Real3> local_coords;
  for (Integer i = 0; i < nb_elements; ++i) {
    Integer current_cell_nb_node = mesh_info.cells_nb_node.item(i);
    Integer cell_type = mesh_info.cells_type.item(i);
    Int64 cell_uid = i;
    cells_infos.add(cell_type);
    cells_infos.add(cell_uid); //cell_unique_id

    ArrayView<Int64> local_info(current_cell_nb_node,&mesh_info.cells_connectivity[connectivity_index]);
    cells_infos.addRange(local_info);
    connectivity_index += current_cell_nb_node;
  }

  IPrimaryMesh* pmesh = mesh->toPrimaryMesh();
  info() << "## Allocating ##";

  if (is_parallel && sid != 0)
    pmesh->allocateCells(0, UniqueArray<Int64>(0), false);
  else
    pmesh->allocateCells(nb_elements, cells_infos, false);

  info() << "## Ending ##";
  pmesh->endAllocate();
  info() << "## Done ##";

  // Positionne les coordonnées
  {
    VariableNodeReal3& nodes_coord_var(pmesh->nodesCoordinates());
    bool has_map = mesh_info.node_coords.empty();
    if (has_map){
      ENUMERATE_NODE (i, mesh->ownNodes()) {
        Node node = *i;
        nodes_coord_var[node] = mesh_info.node_coords_map.lookupValue(node.uniqueId().asInt64());
      }
      nodes_coord_var.synchronize();
    }
    else {
      ENUMERATE_NODE (node, mesh->allNodes()) {
        nodes_coord_var[node] = mesh_info.node_coords.item(node->uniqueId().asInt32());
      }
    }
    //ENUMERATE_NODE(node,mesh->allNodes()) info() << "["<<nodes_coord_var[node].x<<","<<nodes_coord_var[node].y<<","<<nodes_coord_var[node].z<<"]";
  }
  return RTOk;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 *
 The version 2.0 of the '.msh' file format is Gmsh's new native mesh file format. It is very
 similar to the old one (see Section 9.1.1 [Version 1.0], page 139), but is more general: it
 contains information about itself and allows to associate an arbitrary number of integer tags
 with each element. Ialso exists in both ASCII and binary form.
 The '.msh' file format, version 2.0, is divided in three main sections, defining the file
 format ($MeshFormat-$EndMeshFormat), the nodes ($Nodes-$EndNodes) and the elements
 ($Elements-$EndElements) in the mesh:
 /code
 * $MeshFormat
 * 2.0 file-type data-size
 *               (one-binary) is an integer of value 1 written in binary form.
 *               This integer is used for detecting if the computer
 *               on which the binary file was written and the computer
 *               on which the file is read are of the same type
 *               (little or big endian).
 *
 * $EndMeshFormat
 * $Nodes
 * number-of-nodes
 * node-number x-coord y-coord z-coord
 * ...
 * $EndNodes
 * $Elements
 * number-of-elements
 * elm-number elm-type number-of-tags < tag > ... node-number-list
 * ...	
 * $EndElements
 \endcode
*/
IMeshReader::eReturnType MshMeshReader::
_readMeshFromNewMshFile(IMesh* mesh, IosFile& ios_file)
{
  const char* func_name = "MshMeshReader::_readMeshFromNewMshFile()";
  info() << "[_readMeshFromNewMshFile] New native mesh file format detected";
  MeshInfo mesh_info;
  //UniqueArray<Int64> cells_infos;
  //UniqueArray<Integer> cells_nb_node;
  //UniqueArray<Int64> cells_connectivity;
  //UniqueArray<Integer> cells_type;
#define MSH_BINARY_TYPE 1

  Real version = ios_file.getReal();
  if (version==2.0)
    m_version = 2;
  else if (version==4.1)
    m_version = 4;
  else
    ARCANE_THROW(IOException, "Wrong msh file version '{0}'. Only versions '2.0' or '4.1' are supported",version);
  info() << "Msh mesh_major_version=" << m_version;
  Integer file_type = ios_file.getInteger(); // is an integer equal to 0 in the ASCII file format, equal to 1 for the binary format
  if (file_type == MSH_BINARY_TYPE)
    ARCANE_THROW(IOException, "Binary mode is not supported!");

  Integer data_size = ios_file.getInteger(); // is an integer equal to the size of the floating point numbers used in the file
  ARCANE_UNUSED(data_size);

  if (file_type == MSH_BINARY_TYPE) {
    (void)ios_file.getInteger(); // is an integer of value 1 written in binary form
  }
  ios_file.getNextLine(); // Skip current \n\r

  // $EndMeshFormat
  if (!ios_file.lookForString("$EndMeshFormat"))
    ARCANE_THROW(IOException, "$EndMeshFormat not found");

  // TODO: Les différentes sections ($Nodes, $Entitites, ...) peuvent
  // être dans n'importe quel ordre (à part $Nodes qui doit être avant $Elements)
  // Faire un méthode qui gère cela.

  // Après le format, on peut avoir les entités mais cela est optionnel
  // Si elles sont présentes, on lit le fichier jusqu'à la fin de cette section.
  StringView next_line = ios_file.getNextLine();
  if (next_line=="$Entities"){
    do { next_line = ios_file.getNextLine(); } while (next_line != "$EndEntities");
    next_line = ios_file.getNextLine();
  }
  // $Nodes
  if (next_line!="$Nodes")
    ARCANE_THROW(IOException,"Unexpected string '{0}'. Valid values are '$Nodes'",next_line);

  // Fetch nodes number and the coordinates
  if (file_type != MSH_BINARY_TYPE) {
    if (m_version==2){
      if (_readNodesFromAsciiMshV2File(ios_file, mesh_info.node_coords) != RTOk)
        ARCANE_THROW(IOException, "Ascii nodes coords error");
    }
    else if (m_version==4){
      if (_readNodesFromAsciiMshV4File(ios_file, mesh_info) != RTOk)
        ARCANE_THROW(IOException, "Ascii nodes coords error");
    }
  }
  else {
    if (_readNodesFromBinaryMshFile(ios_file, mesh_info.node_coords) != RTOk)
      ARCANE_THROW(IOException, "Binary nodes coords error");
  }
  // $EndNodes
  if (!ios_file.lookForString("$EndNodes"))
    ARCANE_THROW(IOException, "$EndNodes not found");

  // $Elements
  Integer mesh_dimension = -1;
  {
    if (!ios_file.lookForString("$Elements"))
      ARCANE_THROW(IOException, "$Elements not found");

    if (m_version==2)
      mesh_dimension = _readElementsFromAsciiMshV2File(ios_file,mesh_info);
    else if (m_version==4)
      mesh_dimension = _readElementsFromAsciiMshV4File(ios_file,mesh_info);

    // $EndElements
    if (!ios_file.lookForString("$EndElements"))
      throw IOException(func_name, "$EndElements not found");
  }

  info() << "Computed mesh dimension = " << mesh_dimension;

  IPrimaryMesh* pmesh = mesh->toPrimaryMesh();
  pmesh->setDimension(mesh_dimension);

  return _createMeshFrom(mesh, mesh_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
	readMeshFromMshFile switch wether the targetted file is to be read with
	_readMeshFromOldMshFile or _readMeshFromNewMshFile function.
*/
IMeshReader::eReturnType MshMeshReader::
_readMeshFromMshFile(IMesh* mesh, const XmlNode& mesh_node,
                    const String& filename, bool use_internal_partition)
{
  ARCANE_UNUSED(use_internal_partition);
  ARCANE_UNUSED(mesh_node);

  info() << "Trying to read 'msh' file '" << filename << "'";
  ifstream ifile(filename.localstr());
  if (!ifile) {
    error() << "Unable to read file '" << filename << "'";
    return RTError;
  }
  IosFile ios_file(&ifile);
  String mesh_format_str = ios_file.getNextLine(); // Comments do not seem to be implemented in .msh files
  if (IosFile::isEqualString(mesh_format_str, "$MeshFormat"))
    return _readMeshFromNewMshFile(mesh, ios_file);
  info() << "The file does not begin with '$MeshFormat' returning RTError";
  return RTError;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
