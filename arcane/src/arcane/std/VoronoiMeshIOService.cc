// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VoronoiMeshIOService.cc                                     (C) 2000-2009 */
/*                                                                           */
/* Lecture/Ecriture d'un maillage voronoi. Format provisoire.                */
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

#include "arcane/utils/Real3.h"

#include "arcane/FactoryService.h"
#include "arcane/IMeshReader.h"
#include "arcane/ISubDomain.h"
#include "arcane/IMeshSubMeshTransition.h"
#include "arcane/IItemFamily.h"
#include "arcane/Item.h"
#include "arcane/ItemEnumerator.h"
#include "arcane/VariableTypes.h"
#include "arcane/IVariableAccessor.h"
#include "arcane/IParallelMng.h"
#include "arcane/IIOMng.h"
#include "arcane/IMeshUtilities.h"
#include "arcane/IMeshWriter.h"
#include "arcane/BasicService.h"
#include "arcane/XmlNode.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VoronoiFile;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Lecteur des fichiers de maillage au format Voronoi interne IFP.
 *
 * Il s'agit d'une version préliminaire en attendant un format IFP unifié,
 * le lecteur et l'écrivain n'ont été que partiellement testés.
 *
 * Il est possible de spécifier un ensemble de variables dans le fichier.
 * Dans ce cas, leurs valeurs sont lues en même temps que le maillage
 * et servent à initialiser les variables.
 *
 */
class VoronoiMeshIOService
: public BasicService
, public IMeshReader
{
public:

  VoronoiMeshIOService(const ServiceBuildInfo& sbi);

  ~VoronoiMeshIOService();

public:

  virtual void build() {}

public:

  virtual bool allowExtension(const String& str)
  {
    return str=="vor";
  }

 public:

  virtual eReturnType readMeshFromFile(IPrimaryMesh* mesh,const XmlNode& mesh_node,const String& file_name,
                                       const String& dir_name,bool use_internal_partition);

  virtual bool hasCutInfos() const { return false; }
  virtual ConstArrayView<Integer> communicatingSubDomains() const { return ConstArrayView<Integer>(); }

 private:

  bool _readMesh(IPrimaryMesh* mesh,const String& file_name,const String& dir_name,bool use_internal_partition);
  bool _readHybridGrid(IPrimaryMesh* mesh,VoronoiFile& voronoi_file,bool use_internal_partition);
  void _readCellVariable(IMesh* mesh,VoronoiFile& voronoi_file,const String& name_str,Integer nb_cell);
  void _readItemGroup(IMesh* mesh,VoronoiFile& voronoi_file,const String& name_str,Integer nb_item,
                      eItemKind ik,ConstArrayView<Integer> local_ids);
  void _createFaceGroup(IMesh* mesh,const String& name,ConstArrayView<Integer> faces_lid);
  bool _readData(IMesh* mesh,VoronoiFile& voronoi_file,bool use_internal_partition,eItemKind cell_kind,
  		ConstArrayView<Integer> local_ids);
  void _readNodesHybridGrid(IMesh* mesh,VoronoiFile& voronoi_file,Array<Real3>& node_coords);
  void _readCellsHybridGrid(IMesh* mesh,VoronoiFile& voronoi_file,
                                  Array<Integer>& cells_nb_node,
                                  Array<Integer>& cells_type,
                                  Array<Int64>& cells_connectivity,
                                  Integer& mesh_dimension);
  void _readFacesMesh(IMesh* mesh,const String& file_name,
                      const String& dir_name,bool use_internal_partition);

private:
  //! Table des variables crées localement par lecture du maillage
  UniqueArray<VariableCellReal3 *> m_variables;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VoronoiFile
{
 public:
  static const int BUFSIZE = 10000;
 public:
  VoronoiFile(std::istream* stream) : m_stream(stream) {}
  const char* getNextLine();
  Real getReal();
  Integer getInteger();
  bool isEnd(){ (*m_stream) >> ws; return m_stream->eof(); }
 private:
  std::istream* m_stream;
  char m_buf[BUFSIZE];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const char* VoronoiFile::
getNextLine()
{
  while (m_stream->good()){
    m_stream->getline(m_buf,sizeof(m_buf)-1);
    if (m_stream->eof())
      break;
    bool is_comment = true;
    if (m_buf[0]=='\n' || m_buf[0]=='\r')
      continue;
    // Regarde si un caractère de commentaire est présent
    for( int i=0; i<BUFSIZE && m_buf[i]!='\0'; ++i ){
      if (!isspace(m_buf[i])){
        is_comment = (m_buf[i]=='#');
        break;
      }
    }
    if (!is_comment){

      // Supprime le '\n' ou '\r' final
      for( int i=0; i<BUFSIZE && m_buf[i]!='\0'; ++i ){
        //cout << " V=" << m_buf[i] << " I=" << (int)m_buf[i] << "\n";
        if (m_buf[i]=='\n' || m_buf[i]=='\r'){
          m_buf[i] = '\0';
          break;
        }
      }
      return m_buf;
    }
  }
  throw IOException("VoronoiFile::getNexLine()","Unexpected EndOfFile");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real VoronoiFile::
getReal()
{
  Real v = 0.;
  (*m_stream) >> ws >> v;
  if (m_stream->good())
    return v;
  throw IOException("VoronoiFile::getReal()","Bad Real");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer VoronoiFile::
getInteger()
{
  Integer v = 0;
  (*m_stream) >> ws >> v;
  if (m_stream->good())
    return v;
  throw IOException("VoronoiFile::getInteger()","Bad Integer");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SUB_DOMAIN_FACTORY(VoronoiMeshIOService,IMeshReader,VoronoiMeshIO);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VoronoiMeshIOService::
VoronoiMeshIOService(const ServiceBuildInfo& sbi)
: BasicService(sbi)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VoronoiMeshIOService::
~VoronoiMeshIOService()
{
  const Integer size = m_variables.size();
  for(Integer i=0;i<size;++i)
    {
      delete m_variables[i];
      m_variables[i] = NULL;
    }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \todo Verifier plantage sous linux.
 */
IMeshReader::eReturnType VoronoiMeshIOService::
readMeshFromFile(IPrimaryMesh* mesh,const XmlNode& mesh_node,
                 const String& filename,const String& dir_name,
                 bool use_internal_partition)
{
  ARCANE_UNUSED(mesh_node);
  ARCANE_UNUSED(use_internal_partition);

  bool ret = _readMesh(mesh,filename,dir_name,use_internal_partition);
  if (ret)
    return RTError;

  return RTOk;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool VoronoiMeshIOService::
_readMesh(IPrimaryMesh* mesh,const String& file_name,const String& dir_name,
          bool use_internal_partition)
{
  ARCANE_UNUSED(dir_name);

  std::ifstream ifile(file_name.localstr());
  if (!ifile){
    error() << "Unable to read file '" << file_name << "'";
    return true;
  }
  VoronoiFile voronoi_file(&ifile);

  // Lecture du type de maillage
  // TODO: en parallèle, avec use_internal_partition vrai, seul le processeur 0
  // lit les données. Dans ce cas, inutile que les autres ouvre le fichier.

  bool ret = true;
  ret = _readHybridGrid(mesh,voronoi_file,use_internal_partition);
  return ret;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/



/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Lecture des noeuds et de leur coordonnées.
 */
void VoronoiMeshIOService::
_readNodesHybridGrid(IMesh* mesh,VoronoiFile& voronoi_file,Array<Real3>& node_coords)
{
  ARCANE_UNUSED(mesh);

  const char* func_name = "VoronoiMeshIOService::_readNodesHybridGrid()";
  const char* buf = voronoi_file.getNextLine();
  std::istringstream iline(buf);
  std::string points_str;
  std::string data_type_str;
  Integer nb_node = 0;
  iline >> ws >> nb_node;
  if (!iline)
    throw IOException(func_name,"Syntax error while reading number of nodes");
  if (nb_node<0){
    String msg = String("Invalid number of nodes: n=") + nb_node;
    throw IOException(func_name,msg);
  }

  info() << " Info: " << nb_node;

  // Lecture les coordonnées
  node_coords.resize(nb_node);
  {
    for( Integer i=0; i<nb_node; ++i ){
      Real nx = voronoi_file.getReal();
      Real ny = voronoi_file.getReal();
      Real nz = voronoi_file.getReal();
      node_coords[i] = Real3(nx,ny,nz);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Lecture des mailles et de leur connectivité.
 *
 * En retour, remplit \a cells_nb_node, \a cells_type et \a cells_connectivity.
 */
void VoronoiMeshIOService::
_readCellsHybridGrid(IMesh* mesh,VoronoiFile& voronoi_file,
                     Array<Integer>& cells_nb_node,
                     Array<Integer>& cells_type,
                     Array<Int64>& cells_connectivity,
                     Integer& mesh_dimension)
{
  ARCANE_UNUSED(mesh);

  const char* func_name = "VoronoiMeshIOService::_readCellsHybridGrid()";
  const char* buf = voronoi_file.getNextLine();
  std::istringstream iline(buf);
  std::string cells_str;
  Integer nb_cell = 0;
  Integer nb_cell_node = 0;
  iline >> ws >> nb_cell >> ws >> nb_cell_node;
  if (!iline)
    throw IOException(func_name,"Syntax error while reading cells");
  if (nb_cell<0 || nb_cell_node<0){
    throw IOException(func_name,String::format("Invalid dimensions: nb_cell={0} nb_cell_node={1}",nb_cell,nb_cell_node));
  }

  ItemTypeMng * item_type_mng = mesh->itemTypeMng();

  mesh_dimension = 3; // will be set to two if for all types, nb_edges == nb_faces
  bool is_mesh_2d = true;

  cells_nb_node.resize(nb_cell);
  cells_type.resize(nb_cell);
  cells_connectivity.resize(nb_cell_node);
  {
    Integer connectivity_index = 0;
    for( Integer i=0; i<nb_cell; ++i ){
      const Integer user_cell_type = voronoi_file.getInteger();
      const Integer internal_cell_type = user_cell_type + ItemTypeMng::nbBuiltInItemType();
      cells_type[i] = internal_cell_type;
      if (user_cell_type < 0 || internal_cell_type >= ItemTypeMng::nbBasicItemType())
        fatal() << "Bad item type " << user_cell_type;
      /* const Integer n_check = */ voronoi_file.getInteger();
      ItemTypeInfo* item_type = item_type_mng->typeFromId(internal_cell_type);
      is_mesh_2d = is_mesh_2d && (item_type->nbLocalEdge() == item_type->nbLocalFace());
      // optimisable avec un cache local
      const Integer n = item_type->nbLocalNode();
      cells_nb_node[i] = n;
      for( Integer j=0; j<n; ++j ){
        const Integer node_id = voronoi_file.getInteger();
        cells_connectivity[connectivity_index] = node_id;
        ++connectivity_index;
      }
    }
  }
  if (is_mesh_2d) mesh_dimension = 2;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool VoronoiMeshIOService::
_readHybridGrid(IPrimaryMesh* mesh,VoronoiFile& voronoi_file,bool use_internal_partition)
{
  // const char* func_name = "VoronoiMeshIOService::_readUnstructuredGrid()";
  //IParallelMng* pm = subDomain()->parallelMng();
  Integer nb_cell = 0;
  Integer nb_cell_node = 0;
  Integer sid = subDomain()->subDomainId();
  UniqueArray<Real3> node_coords;
  UniqueArray<Int64> cells_infos;
  UniqueArray<Integer> cells_local_id;
  UniqueArray<Integer> faces_local_id;
  // Si on utilise le partitionneur interne, seul le sous-domaine lit le maillage
  bool need_read = true;
  Integer mesh_dimension(-1);
  if (use_internal_partition)
    need_read = (sid==0);

  if (need_read){
    _readNodesHybridGrid(mesh,voronoi_file,node_coords);
    //nb_node = node_coords.size();

    // Lecture des infos des mailles
    // Lecture de la connectivité
    UniqueArray<Integer> cells_nb_node;
    UniqueArray<Int64> cells_connectivity;
    UniqueArray<Integer> cells_type;
    _readCellsHybridGrid(mesh,voronoi_file,cells_nb_node,cells_type,cells_connectivity, mesh_dimension);
    nb_cell = cells_nb_node.size();
    nb_cell_node = cells_connectivity.size();
    cells_local_id.resize(nb_cell);

    // Création des mailles
    // Infos pour la création des mailles
    // par maille: 1 pour son unique id,
    //             1 pour son type,
    //             1 pour chaque noeud
    cells_infos.resize(nb_cell*2 + nb_cell_node);
    {
      Integer cells_infos_index = 0;
      Integer connectivity_index = 0;
      for( Integer i=0; i<nb_cell; ++i ){
        Integer current_cell_nb_node = cells_nb_node[i];
        Integer cell_unique_id = i;

        cells_local_id[i] = i;

        cells_infos[cells_infos_index] = cells_type[i];
        ++cells_infos_index;

        cells_infos[cells_infos_index] = cell_unique_id;
        ++cells_infos_index;

        for( Integer z=0; z<current_cell_nb_node; ++z ){
          cells_infos[cells_infos_index+z] = cells_connectivity[connectivity_index+z];
        }
        cells_infos_index += current_cell_nb_node;
        connectivity_index += current_cell_nb_node;
      }
    }

  }

  Integer dimension = subDomain()->parallelMng()->reduce(Parallel::ReduceMax,mesh_dimension);
  mesh->setDimension(dimension);
  mesh->allocateCells(nb_cell,cells_infos,false);
  mesh->endAllocate();

  // Positionne les coordonnées
  {
    VariableNodeReal3& nodes_coord_var(mesh->nodesCoordinates());
    ENUMERATE_NODE(inode,mesh->allNodes()){
      Node node = *inode;
      Int32 node_uid = node.uniqueId().asInt32();
      nodes_coord_var[inode] = node_coords[node_uid];
    }
  }

  // Maintenant, regarde s'il existe des données associées aux fichier
  bool r = _readData(mesh,voronoi_file,use_internal_partition,IK_Cell,cells_local_id);
  if (r)
    return r;

  return false;
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool VoronoiMeshIOService::
_readData(IMesh* mesh,VoronoiFile& voronoi_file,bool use_internal_partition,
          eItemKind cell_kind,ConstArrayView<Integer> local_ids)
{
  ARCANE_UNUSED(use_internal_partition);
  ARCANE_UNUSED(local_ids);

	// Seul le sous-domain maitre lit les valeurs. Par contre, les autres
	// sous-domaines doivent connaitre la liste des variables
	Integer sid = subDomain()->subDomainId();
	Integer nb_cell_kind = mesh->nbItem(cell_kind);
	if (sid==0){
		_readCellVariable(mesh,voronoi_file,"CellCenter",nb_cell_kind);
	}
	if (sid!=0){
		 VariableCellReal3 * var = new VariableCellReal3(VariableBuildInfo(mesh,"CellCenter"));
		 m_variables.add(var);
	}

  return false;
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VoronoiMeshIOService::
_readCellVariable(IMesh* mesh,VoronoiFile& voronoi_file,const String& var_name,Integer nb_cell)
{
  //TODO Faire la conversion uniqueId() vers localId() correcte
  Real cx,cy,cz;
  info() << "Reading values for variable: " << var_name << " n=" << nb_cell;
  VariableCellReal3 * var = new VariableCellReal3(VariableBuildInfo(mesh,var_name));
  m_variables.add(var);
  Real3ArrayView values(var->asArray());
  for( Integer i=0; i<nb_cell; ++i ){
    cx  = voronoi_file.getReal();
    cy  = voronoi_file.getReal();
    cz  = voronoi_file.getReal();
    values[i] = Real3(cx,cy,cz);
  }
  info() << "Variable build done: " << voronoi_file.isEnd();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VoronoiMeshIOService::
_readItemGroup(IMesh* mesh,VoronoiFile& voronoi_file,const String& name,Integer nb_item,
               eItemKind ik,ConstArrayView<Integer> local_ids)
{
  IItemFamily* item_family = mesh->itemFamily(ik);
  info() << "Reading group inf for group: " << name;

  IntegerUniqueArray ids;
  for( Integer i=0; i<nb_item; ++i ){
    Integer v = voronoi_file.getInteger();
    if (v!=0)
      ids.add(local_ids[i]);
  }
  info() << "Building group: " << name << " nb_element=" << ids.size();

  item_family->createGroup(name,ids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
