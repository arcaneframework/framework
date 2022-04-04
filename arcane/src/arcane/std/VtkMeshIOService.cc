﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VtkMeshIOService.cc                                         (C) 2000-2020 */
/*                                                                           */
/* Lecture/Ecriture d'un maillage au format Vtk historique (legacy).         */
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
#include "arcane/utils/OStringStream.h"

#include "arcane/utils/Real3.h"

#include "arcane/FactoryService.h"
#include "arcane/ICaseMeshReader.h"
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
#include "arcane/IXmlDocumentHolder.h"
#include "arcane/XmlNodeList.h"
#include "arcane/XmlNode.h"
#include "arcane/IMeshUtilities.h"
#include "arcane/IMeshWriter.h"
#include "arcane/BasicService.h"
#include "arcane/IMeshBuilder.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VtkFile;

namespace
{
const int VTK_EMPTY_CELL = 0;
const int VTK_VERTEX = 1;
const int VTK_LINE  = 3;
const int VTK_TRIANGLE = 5;
const int VTK_POLYGON = 7; // A tester...
const int VTK_QUAD =  9;
const int VTK_TETRA = 10;
const int VTK_HEXAHEDRON = 12;
const int VTK_WEDGE = 13;
const int VTK_PYRAMID = 14;
const int VTK_PENTAGONAL_PRISM = 15;
const int VTK_HEXAGONAL_PRISM = 16;
const int VTK_QUADRATIC_EDGE =  21;
const int VTK_QUADRATIC_TRIANGLE =  22;
const int VTK_QUADRATIC_QUAD =  23;
const int VTK_QUADRATIC_TETRA =  24;
const int VTK_QUADRATIC_HEXAHEDRON =  25;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Lecteur des fichiers de maillage au format Vtk historique (legacy).
 *
 * Il s'agit d'une version préliminaire qui ne supporte que les
 * DATASET de type STRUCTURED_GRID ou UNSTRUCTURED_GRID. De plus,
 * le lecteur et l'écrivain n'ont été que partiellement testés.
 *
 * L'en-tête du fichier vtk doit être:
 * # vtk DataFile Version 2.0
 *
 * Il est possible de spécifier un ensemble de variables dans le fichier.
 * Dans ce cas, leurs valeurs sont lues en même temps que le maillage
 * et servent à initialiser les variables. Actuellement, seules les valeurs
 * aux mailles sont supportées
 *
 * Comme Vtk ne supporte pas la notion de groupe, il est possible
 * de spécifier un groupe comme étant une variable (CELL_DATA).
 * Par convention, si la variable commence par la chaine 'GROUP_', alors
 * il s'agit d'un groupe. La variable doit être déclarée comme suit:
 * \begincode
 * CELL_DATA %n
 * SCALARS GROUP_%m int 1
 * LOOKUP_TABLE default
 * \endcode
 * avec %n le nombre de mailles, et %m le nom du groupe.
 * Une maille appartient au groupe si la valeur de la donnée est
 * différente de 0.
 *
 * Actuellement, on NE peut PAS spécifier de groupes de points.
 *
 * Pour spécifier des groupes de faces, il faut un fichier vtk
 * additionnel, identique au fichier d'origine mais contenant la
 * description des faces au lieu des mailles. Par convention, si le
 * fichier courant lu s'appelle 'toto.vtk', le fichier décrivant les
 * faces sera 'toto.vtkfaces.vtk'. Ce fichier est optionnel.
 */
class VtkMeshIOService
: public TraceAccessor
{
 public:

  explicit VtkMeshIOService(ITraceMng* tm)
  : TraceAccessor(tm){}
  ~VtkMeshIOService();

 public:

  void build() {}

 public:

  enum eMeshType
  {
    VTK_MT_Unknown,
    VTK_MT_StructuredGrid,
    VTK_MT_UnstructuredGrid
  };

  class VtkMesh
  {
   public:
  };

  class VtkStructuredGrid
  : public VtkMesh
  {
  public:
    int m_nb_x;
    int m_nb_y;
    int m_nb_z;
  };

 public:
  
  bool readMesh(IPrimaryMesh* mesh,const String& file_name,const String& dir_name,bool use_internal_partition);
 private:

  bool _readStructuredGrid(IPrimaryMesh* mesh,VtkFile&,bool use_internal_partition);
  bool _readUnstructuredGrid(IPrimaryMesh* mesh,VtkFile& vtk_file,bool use_internal_partition);
  void _readCellVariable(IMesh* mesh,VtkFile& vtk_file,const String& name_str,Integer nb_cell);
  void _readItemGroup(IMesh* mesh,VtkFile& vtk_file,const String& name_str,Integer nb_item,
                      eItemKind ik,ConstArrayView<Int32> local_ids);
  void _readNodeGroup(IMesh* mesh,VtkFile& vtk_file,const String& name,Integer nb_item);
  void _createFaceGroup(IMesh* mesh,const String& name,Int32ConstArrayView faces_lid);
  bool _readData(IMesh* mesh,VtkFile& vtk_file,bool use_internal_partition,eItemKind cell_kind,
                 Int32ConstArrayView local_ids,Integer nb_node);
  void _readNodesUnstructuredGrid(IMesh* mesh,VtkFile& vtk_file,Array<Real3>& node_coords);
  void _readCellsUnstructuredGrid(IMesh* mesh,VtkFile& vtk_file,
                                  Array<Integer>& cells_nb_node,
                                  Array<Integer>& cells_type,
                                  Array<Int64>& cells_connectivity);
  void _readFacesMesh(IMesh* mesh,const String& file_name,
                      const String& dir_name,bool use_internal_partition);

 private:

  //! Table des variables crées localement par lecture du maillage
  UniqueArray<VariableCellReal*> m_variables;

 private:

};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VtkFile
{
 public:
  static const int BUFSIZE = 10000;
 public:
  VtkFile(std::istream* stream) : m_stream(stream) {}
  const char* getNextLine();
  Real getReal();
  Integer getInteger();
  void checkString(const String& current_value,const String& expected_value);
  void checkString(const String& current_value,
                   const String& expected_value1,
                   const String& expected_value2);
  static bool isEqualString(const String& current_value,const String& expected_value);

  bool isEnd(){ (*m_stream) >> ws; return m_stream->eof(); }
 private:
  std::istream* m_stream;
  char m_buf[BUFSIZE];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const char* VtkFile::
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
  throw IOException("VtkFile::getNexLine()","Unexpected EndOfFile");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real VtkFile::
getReal()
{
  Real v = 0.;
  (*m_stream) >> ws >> v;
  if (m_stream->good())
    return v;
  throw IOException("VtkFile::getReal()","Bad Real");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer VtkFile::
getInteger()
{
  Integer v = 0;
  (*m_stream) >> ws >> v;
  if (m_stream->good())
    return v;
  throw IOException("VtkFile::getInteger()","Bad Integer");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkFile::
checkString(const String& current_value,const String& expected_value)
{
  String current_value_low = current_value.lower(); 
  String expected_value_low = expected_value.lower(); 
 
  if (current_value_low!=expected_value_low){
    String s = "Expecting chain '" + expected_value + "', found '" + current_value + "'";
    throw IOException("VtkFile::checkString()",s);
  }
}

void VtkFile::
checkString(const String& current_value,const String& expected_value1,const String& expected_value2)
{
  String current_value_low = current_value.lower(); 
  String expected_value1_low = expected_value1.lower();
  String expected_value2_low = expected_value2.lower(); 

  if (current_value_low!=expected_value1_low && current_value_low!=expected_value2_low){
    String s = "Expecting chain '" + expected_value1 + "' or '"
      + expected_value2 + "', found '" + current_value + "'";
    throw IOException("VtkFile::checkString()",s);
  }
}

bool VtkFile::
isEqualString(const String& current_value,const String& expected_value)
{
  String current_value_low = current_value.lower(); 
  String expected_value_low = expected_value.lower(); 
  return (current_value_low==expected_value_low);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VtkMeshIOService::
~VtkMeshIOService()
{
  const Integer size = m_variables.size();
  for( Integer i=0;i<size;++i) {
    delete m_variables[i];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool VtkMeshIOService::
readMesh(IPrimaryMesh* mesh,const String& file_name,const String& dir_name,bool use_internal_partition)
{
  std::ifstream ifile(file_name.localstr());
  if (!ifile){
    error() << "Unable to read file '" << file_name << "'";
    return true;
  }
  VtkFile vtk_file(&ifile);
  const char* buf = 0;
  // Lecture de la description
  buf = vtk_file.getNextLine();
  String format = vtk_file.getNextLine();
  if (! VtkFile::isEqualString(format,"ASCII")){
    error() << "Support exists only for 'ASCII' format (format='" << format << "')";
    return true;
  }
  eMeshType mesh_type = VTK_MT_Unknown;
  // Lecture du type de maillage
  // TODO: en parallèle, avec use_internal_partition vrai, seul le processeur 0
  // lit les données. Dans ce cas, inutile que les autres ouvre le fichier.
  {
    buf = vtk_file.getNextLine();
    std::istringstream mesh_type_line(buf);
    std::string dataset_str;
    std::string mesh_type_str;
    mesh_type_line >> ws >> dataset_str >> ws >> mesh_type_str;
    vtk_file.checkString(dataset_str,"DATASET");
    if (VtkFile::isEqualString(mesh_type_str,"STRUCTURED_GRID")){
      mesh_type = VTK_MT_StructuredGrid;
    }
    if (VtkFile::isEqualString(mesh_type_str,"UNSTRUCTURED_GRID")){
      mesh_type = VTK_MT_UnstructuredGrid;
    }
    if (mesh_type==VTK_MT_Unknown){
      error() << "Support exists only for 'STRUCTURED_GRID' and 'UNSTRUCTURED_GRID' formats (format=" << mesh_type_str << "')";
      return true;
    }
  }
  bool ret = true;
  switch(mesh_type){
  case VTK_MT_StructuredGrid:
    ret = _readStructuredGrid(mesh,vtk_file,use_internal_partition);
    break;
  case VTK_MT_UnstructuredGrid:
    ret = _readUnstructuredGrid(mesh,vtk_file,use_internal_partition);
    if (!ret){
      // Tente de lire le fichier des faces s'il existe
      _readFacesMesh(mesh,file_name+"faces.vtk",dir_name,use_internal_partition);
    }
    break;
  case VTK_MT_Unknown:
    break;
  }
  /*while ( (buf=vtk_file.getNextLine()) != 0 ){
    info() << " STR " << buf;
    }*/
  return ret;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool VtkMeshIOService::
_readStructuredGrid(IPrimaryMesh* mesh,VtkFile& vtk_file,bool use_internal_partition)
{
  // Lecture du nombre de points: DIMENSIONS nx ny nz
  const char* buf = nullptr;
  Integer nb_node_x = 0;
  Integer nb_node_y = 0;
  Integer nb_node_z = 0;
  {
    buf = vtk_file.getNextLine();
    std::istringstream iline(buf);
    std::string dimension_str;
    iline >> ws >> dimension_str >> ws >> nb_node_x
          >> ws >> nb_node_y >> ws >> nb_node_z;
    if (!iline){
      error() << "Syntax error while reading grid dimensions";
      return true;
    }
    vtk_file.checkString(dimension_str,"DIMENSIONS");
    if (nb_node_x<=1 || nb_node_y<=1 || nb_node_z<=1){
      error() << "Invalid dimensions: x=" << nb_node_x << " y=" << nb_node_y << " z=" << nb_node_z;
      return true;
    }
  }
  info() << " Infos: " << nb_node_x << " " << nb_node_y << " " << nb_node_z;
  Integer nb_node = nb_node_x * nb_node_y * nb_node_z;
  // Lecture du nombre de points: POINTS nb float
  {
    buf = vtk_file.getNextLine();
    std::istringstream iline(buf);
    std::string points_str;
    std::string float_str;
    Integer nb_node_read = 0;
    iline >> ws >> points_str >> ws >> nb_node_read
          >> ws >> float_str;
    if (!iline){
      error() << "Syntax error while reading grid dimensions";
      return true;
    }
    vtk_file.checkString(points_str,"POINTS");
    vtk_file.checkString(float_str,"float");
    if (nb_node_read!=nb_node){
      error() << "Number of invalid nodes: expected=" << nb_node << " found=" << nb_node_read;
      return true;
    }
  }

  Int32 sub_domain_id = mesh->parallelMng()->commRank();

  Integer nb_cell_x = nb_node_x-1;
  Integer nb_cell_y = nb_node_y-1;
  Integer nb_cell_z = nb_node_z-1;

  if (use_internal_partition && sub_domain_id!=0){
    nb_node_x = 0;
    nb_node_y = 0;
    nb_node_z = 0;
    nb_cell_x = 0;
    nb_cell_y = 0;
    nb_cell_z = 0;
  }

  const Integer nb_node_yz = nb_node_y*nb_node_z;
  const Integer nb_node_xy = nb_node_x*nb_node_y;

  Integer nb_cell = nb_cell_x * nb_cell_y * nb_cell_z;
  UniqueArray<Int32> cells_local_id(nb_cell);

  // Creation du maillage
  {
    UniqueArray<Integer> nodes_unique_id(nb_node);

    info() << " NODE YZ = " << nb_node_yz;
    // Création des noeuds
    //Integer nb_node_local_id = 0;
    {
      Integer node_local_id = 0;
      for( Integer x=0; x<nb_node_x; ++x ){
        for( Integer z=0; z<nb_node_z; ++z ){
          for( Integer y=0; y<nb_node_y; ++y ){
    
            Integer node_unique_id = y + (z)*nb_node_y + x*nb_node_y*nb_node_z;
          
            nodes_unique_id[node_local_id] = node_unique_id;
            //Integer owner = sub_domain_id;
            
            ++node_local_id;
          }
        }
      }
      //nb_node_local_id = node_local_id;
      //warning() << " NB NODE LOCAL ID=" << node_local_id;
    }

    // Création des mailles
    
    // Infos pour la création des mailles
    // par maille: 1 pour son unique id,
    //             1 pour son type,
    //             8 pour chaque noeud
    UniqueArray<Int64> cells_infos(nb_cell*10);

    {
      Integer cell_local_id = 0;
      Integer cells_infos_index = 0;

      // Normalement ne doit pas arriver car les valeurs de nb_node_x et
      // nb_node_y sont testées lors de la lecture.
      if (nb_node_xy==0)
        ARCANE_FATAL("Null value for nb_node_xy");

      //Integer index = 0;
      for( Integer z=0; z<nb_cell_z; ++z ){
        for( Integer y=0; y<nb_cell_y; ++y ){
          for( Integer x=0; x<nb_cell_x; ++x ){
            Integer current_cell_nb_node = 8;
          
            //Integer cell_unique_id = y + (z)*nb_cell_y + x*nb_cell_y*nb_cell_z;
            Int64 cell_unique_id = x + y*nb_cell_x + z*nb_cell_x*nb_cell_y;
          
            cells_infos[cells_infos_index] = IT_Hexaedron8;
            ++cells_infos_index;

            cells_infos[cells_infos_index] = cell_unique_id;
            ++cells_infos_index;

            //Integer base_id = y + z*nb_node_y + x*nb_node_yz;
            Integer base_id = x + y*nb_node_x + z*nb_node_xy;
            cells_infos[cells_infos_index+0] = nodes_unique_id[base_id];
            cells_infos[cells_infos_index+1] = nodes_unique_id[base_id + 1];
            cells_infos[cells_infos_index+2] = nodes_unique_id[base_id + nb_node_x + 1];
            cells_infos[cells_infos_index+3] = nodes_unique_id[base_id + nb_node_x + 0];
            cells_infos[cells_infos_index+4] = nodes_unique_id[base_id + nb_node_xy];
            cells_infos[cells_infos_index+5] = nodes_unique_id[base_id + nb_node_xy + 1];
            cells_infos[cells_infos_index+6] = nodes_unique_id[base_id + nb_node_xy + nb_node_x + 1];
            cells_infos[cells_infos_index+7] = nodes_unique_id[base_id + nb_node_xy + nb_node_x + 0];
            cells_infos_index += current_cell_nb_node;
            cells_local_id[cell_local_id] = cell_local_id;
            ++cell_local_id;
          }
        }
      }
    }

    mesh->setDimension(3);
    mesh->allocateCells(nb_cell,cells_infos,false);
    mesh->endAllocate();


    // Positionne les coordonnées
    {
      UniqueArray<Real3> coords(nb_node);
      for( Integer z=0; z<nb_node_z; ++z ){
        for( Integer y=0; y<nb_node_y; ++y ){
          for( Integer x=0; x<nb_node_x; ++x ){
            Real nx = vtk_file.getReal();
            Real ny = vtk_file.getReal();
            Real nz = vtk_file.getReal();
            Integer node_unique_id = x + y*nb_node_x + z*nb_node_xy;
            coords[node_unique_id] = Real3(nx,ny,nz);
          }
        }
      }
      VariableNodeReal3& nodes_coord_var(mesh->nodesCoordinates());
      ENUMERATE_NODE(inode,mesh->allNodes()){
        Node node = *inode;
        nodes_coord_var[inode] = coords[node.uniqueId().asInt32()];
      }
    }
  }

  // Créé les groupes de faces des côtés du parallélépipède
  {
    Int32UniqueArray xmin_surface_lid;
    Int32UniqueArray xmax_surface_lid;
    Int32UniqueArray ymin_surface_lid;
    Int32UniqueArray ymax_surface_lid;
    Int32UniqueArray zmin_surface_lid;
    Int32UniqueArray zmax_surface_lid;

    ENUMERATE_FACE(iface,mesh->allFaces()){
      const Face& face = *iface;
      Integer face_local_id = face.localId();
      bool is_xmin = true;
      bool is_xmax = true;
      bool is_ymin = true;
      bool is_ymax = true;
      bool is_zmin = true;
      bool is_zmax = true;
      for( NodeEnumerator inode(face.nodes()); inode(); ++inode ){
        Node node = *inode;
        Int64 node_unique_id = node.uniqueId().asInt64();
        Int64 node_z = node_unique_id / nb_node_xy;
        Int64 node_y = (node_unique_id - node_z*nb_node_xy) / nb_node_x;
        Int64 node_x = node_unique_id - node_z*nb_node_xy - node_y*nb_node_x;
        if (node_x!=0)
          is_xmin = false;
        if (node_x!=(nb_node_x-1))
          is_xmax = false;
        if (node_y!=0)
          is_ymin = false;
        if (node_y!=(nb_node_y-1))
          is_ymax = false;
        if (node_z!=0)
          is_zmin = false;
        if (node_z!=(nb_node_z-1))
          is_zmax = false;
      }
      if (is_xmin)
        xmin_surface_lid.add(face_local_id);
      if (is_xmax)
        xmax_surface_lid.add(face_local_id);
      if (is_ymin)
        ymin_surface_lid.add(face_local_id);
      if (is_ymax)
        ymax_surface_lid.add(face_local_id);
      if (is_zmin)
        zmin_surface_lid.add(face_local_id);
      if (is_zmax)
        zmax_surface_lid.add(face_local_id);
      
    }
    _createFaceGroup(mesh,"XMIN",xmin_surface_lid);
    _createFaceGroup(mesh,"XMAX",xmax_surface_lid);
    _createFaceGroup(mesh,"YMIN",ymin_surface_lid);
    _createFaceGroup(mesh,"YMAX",ymax_surface_lid);
    _createFaceGroup(mesh,"ZMIN",zmin_surface_lid);
    _createFaceGroup(mesh,"ZMAX",zmax_surface_lid);

  }

  // Maintenant, regarde s'il existe des données associées aux fichier
  bool r = _readData(mesh,vtk_file,use_internal_partition,IK_Cell,cells_local_id,nb_node);
  if (r)
    return r;

  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Lecture des noeuds et de leur coordonnées.
 */
void VtkMeshIOService::
_readNodesUnstructuredGrid(IMesh* mesh,VtkFile& vtk_file,Array<Real3>& node_coords)
{
  ARCANE_UNUSED(mesh);

  const char* func_name = "VtkMeshIOService::_readNodesUnstructuredGrid()";
  const char* buf = vtk_file.getNextLine();
  std::istringstream iline(buf);
  std::string points_str;
  std::string data_type_str;
  Integer nb_node = 0;
  iline >> ws >> points_str >> ws >> nb_node >> ws >> data_type_str;
  if (!iline)
    throw IOException(func_name,"Syntax error while reading number of nodes");
  vtk_file.checkString(points_str,"POINTS");
  vtk_file.checkString(data_type_str,"float","double");
  if (nb_node<0)
    throw IOException(A_FUNCINFO,String::format("Invalid number of nodes: n={0}",nb_node));

  info() << " Info: " << nb_node;

  // Lecture les coordonnées
  node_coords.resize(nb_node);
  {
    for( Integer i=0; i<nb_node; ++i ){
      Real nx = vtk_file.getReal();
      Real ny = vtk_file.getReal();
      Real nz = vtk_file.getReal();
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
void VtkMeshIOService::
_readCellsUnstructuredGrid(IMesh* mesh,VtkFile& vtk_file,
                           Array<Integer>& cells_nb_node,
                           Array<Integer>& cells_type,
                           Array<Int64>& cells_connectivity)
{
  ARCANE_UNUSED(mesh);

  const char* func_name = "VtkMeshIOService::_readCellsUnstructuredGrid()";
  const char* buf = vtk_file.getNextLine();
  std::istringstream iline(buf);
  std::string cells_str;
  Integer nb_cell = 0;
  Integer nb_cell_node = 0;
  iline >> ws >> cells_str >> ws >> nb_cell >> ws >> nb_cell_node;
  if (!iline)
    throw IOException(func_name,"Syntax error while reading cells");
  vtk_file.checkString(cells_str,"CELLS");
  if (nb_cell<0 || nb_cell_node<0){
    throw IOException(A_FUNCINFO,
                      String::format("Invalid dimensions: nb_cell={0} nb_cell_node={1}",
                                     nb_cell,nb_cell_node));
  }

  cells_nb_node.resize(nb_cell);
  cells_type.resize(nb_cell);
  cells_connectivity.resize(nb_cell_node);
  {
    Integer connectivity_index = 0;
    for( Integer i=0; i<nb_cell; ++i ){
      Integer n = vtk_file.getInteger();
      cells_nb_node[i] = n;
      for( Integer j=0; j<n; ++j ){
        Integer id = vtk_file.getInteger();
        cells_connectivity[connectivity_index] = id;
        ++connectivity_index;
      }
    }
  }

  // Lecture du type des mailles
  {
    buf = vtk_file.getNextLine();
    std::istringstream iline(buf);
    std::string cell_types_str;
    Integer nb_cell_type;
    iline >> ws >> cell_types_str >> ws >> nb_cell_type;
    if (!iline){
      throw IOException(func_name,"Syntax error while reading cell types");
    }
    vtk_file.checkString(cell_types_str,"CELL_TYPES");
    if (nb_cell_type!=nb_cell){
      throw IOException(A_FUNCINFO,
                        String::format("Inconsistency in number of CELL_TYPES: v={0} nb_cell={1}",
                                       nb_cell_type,nb_cell));
    }
  }
  for( Integer i=0; i<nb_cell; ++i ){
    Integer vtk_ct = vtk_file.getInteger();
    Integer it = IT_NullType;
    // Le type est défini dans vtkCellType.h
    switch(vtk_ct){
    case VTK_EMPTY_CELL: it = IT_NullType; break;
    case VTK_VERTEX: it = IT_Vertex; break;
    case VTK_LINE: it = IT_Line2; break;
    case VTK_QUADRATIC_EDGE: it = IT_Line3; break;
    case VTK_TRIANGLE: it = IT_Triangle3; break;
    case VTK_QUAD: it = IT_Quad4; break;
    case VTK_QUADRATIC_QUAD: it = IT_Quad8; break;
    case VTK_POLYGON: // VTK_POLYGON (a tester...)
      if (cells_nb_node[i]==5)
        it = IT_Pentagon5;
      if (cells_nb_node[i]==6)
        it = IT_Hexagon6;
      break;
    case VTK_TETRA: it = IT_Tetraedron4; break;
    case VTK_QUADRATIC_TETRA: it = IT_Tetraedron10; break;
    case VTK_PYRAMID: it = IT_Pyramid5; break;
    case VTK_WEDGE: it = IT_Pentaedron6; break;
    case VTK_HEXAHEDRON: it = IT_Hexaedron8; break;
    case VTK_QUADRATIC_HEXAHEDRON: it = IT_Hexaedron20; break;
    case VTK_PENTAGONAL_PRISM: it = IT_Heptaedron10; break;
    case VTK_HEXAGONAL_PRISM: it = IT_Octaedron12; break;
      // NOTE GG: les types suivants ne sont pas bon pour VTK.
      //case 27: it = IT_Enneedron14; break; //
      //case 28: it = IT_Decaedron16; break; // VTK_HEXAGONAL_PRISM
      //case 29: it = IT_Heptagon7; break; // VTK_HEPTAGON
      //case 30: it = IT_Octogon8; break; // VTK_OCTAGON
    default:
      ARCANE_THROW(IOException,"Unsupported VtkCellType '{0}'",vtk_ct);
    }
    cells_type[i] = it;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool VtkMeshIOService::
_readUnstructuredGrid(IPrimaryMesh* mesh,VtkFile& vtk_file,bool use_internal_partition)
{
  // const char* func_name = "VtkMeshIOService::_readUnstructuredGrid()";
  //IParallelMng* pm = subDomain()->parallelMng();
  Integer nb_node = 0;
  Integer nb_cell = 0;
  Integer nb_cell_node = 0;
  Int32 sid = mesh->parallelMng()->commRank();
  UniqueArray<Real3> node_coords;
  UniqueArray<Int64> cells_infos;
  UniqueArray<Int32> cells_local_id;
  // Si on utilise le partitionneur interne, seul le sous-domaine lit le maillage
  bool need_read = true;
  if (use_internal_partition)
    need_read = (sid==0);

  bool has_3d_cell = false;

  if (need_read){
    _readNodesUnstructuredGrid(mesh,vtk_file,node_coords);
    nb_node = node_coords.size();

    // Lecture des infos des mailles
    // Lecture de la connectivité
    UniqueArray<Integer> cells_nb_node;
    UniqueArray<Int64> cells_connectivity;
    UniqueArray<Integer> cells_type;
    _readCellsUnstructuredGrid(mesh,vtk_file,cells_nb_node,cells_type,cells_connectivity);
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

    // Regarde si on a au moins une maille 3D. Dans ce cas,
    // le maillage est 3D, sinon il est 2D
    for( Integer i=0; i<nb_cell; ++i ){
      Integer ct = cells_type[i];
      if (ct==IT_Tetraedron4 || ct==IT_Pyramid5 || ct==IT_Pentaedron6 ||
          ct==IT_Hexaedron8 || ct==IT_Heptaedron10 || ct==IT_Octaedron12){
        has_3d_cell = true;
        break;
      }
    }

  }

  // Positionne la dimension du maillage. Comme elle n'est pas indiquée dans
  // le fichier, on regarde si on a au moins une maille 3D.
  // TODO: supporter les maillages 1D
  {
    Integer wanted_dimension = (has_3d_cell) ? 3 : 2;
    wanted_dimension = mesh->parallelMng()->reduce(Parallel::ReduceMax,wanted_dimension);
    mesh->setDimension(wanted_dimension);
  }

  mesh->allocateCells(nb_cell,cells_infos,false);
  mesh->endAllocate();

  // Positionne les coordonnées
  {
    VariableNodeReal3& nodes_coord_var(mesh->nodesCoordinates());
    ENUMERATE_NODE(inode,mesh->allNodes()){
      Node node = *inode;
      nodes_coord_var[inode] = node_coords[node.uniqueId().asInt32()];
    }
  }

  // Maintenant, regarde s'il existe des données associées aux fichier
  bool r = _readData(mesh,vtk_file,use_internal_partition,IK_Cell,cells_local_id,nb_node);
  if (r)
    return r;

  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkMeshIOService::
_readFacesMesh(IMesh* mesh,const String& file_name,const String& dir_name,
               bool use_internal_partition)
{
  ARCANE_UNUSED(dir_name);

  std::ifstream ifile(file_name.localstr());
  if (!ifile){
    info() << "No face descriptor file found '" << file_name << "'";
    return;
  }
  VtkFile vtk_file(&ifile);
  const char* buf = 0;
  // Lecture de la description
  buf = vtk_file.getNextLine();
  String format = vtk_file.getNextLine();
  if (! VtkFile::isEqualString(format,"ASCII")){
    error() << "Support exists only for 'ASCII' format (format='" << format << "')";
    return;
  }
  eMeshType mesh_type = VTK_MT_Unknown;
  // Lecture du type de maillage
  // TODO: en parallèle, avec use_internal_partition vrai, seul le processeur 0
  // lit les données. Dans ce cas, inutile que les autres ouvre le fichier.
  {
    buf = vtk_file.getNextLine();
    std::istringstream mesh_type_line(buf);
    std::string dataset_str;
    std::string mesh_type_str;
    mesh_type_line >> ws >> dataset_str >> ws >> mesh_type_str;
    vtk_file.checkString(dataset_str,"DATASET");
    if (VtkFile::isEqualString(mesh_type_str,"UNSTRUCTURED_GRID")){
      mesh_type = VTK_MT_UnstructuredGrid;
    }
    if (mesh_type==VTK_MT_Unknown){
      error() << "Face descriptor file type must be 'UNSTRUCTURED_GRID' (format=" << mesh_type_str << "')";
      return;
    }
  }
  {
    IParallelMng* pm = mesh->parallelMng();
    Integer nb_face = 0;
    Int32 sid = pm->commRank();
  
    UniqueArray<Int32> faces_local_id;

    // Si on utilise le partitionneur interne, seul le sous-domaine lit le maillage
    bool need_read = true;
    if (use_internal_partition)
      need_read = (sid==0);

    if (need_read){
      {
        // Lit des noeuds, mais ne conserve pas leur coordonnées car cela n'est
        // pas nécessaire.
        UniqueArray<Real3> node_coords;
        _readNodesUnstructuredGrid(mesh,vtk_file,node_coords);
        //nb_node = node_coords.size();
      }

      // Lecture des infos des faces
      // Lecture de la connectivité
      UniqueArray<Integer> faces_nb_node;
      UniqueArray<Int64> faces_connectivity;
      UniqueArray<Integer> faces_type;
      _readCellsUnstructuredGrid(mesh,vtk_file,faces_nb_node,faces_type,faces_connectivity);
      nb_face = faces_nb_node.size();
      //nb_face_node = faces_connectivity.size();
      
      // Il faut à partir de la connectivité retrouver les localId() des faces
      faces_local_id.resize(nb_face);
      {
        IMeshUtilities* mu = mesh->utilities();
        mu->localIdsFromConnectivity(IK_Face,faces_nb_node,faces_connectivity,faces_local_id);
      }
    }
    

    // Maintenant, regarde s'il existe des données associées aux fichiers
    _readData(mesh,vtk_file,use_internal_partition,IK_Face,faces_local_id,0);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool VtkMeshIOService::
_readData(IMesh* mesh,VtkFile& vtk_file,bool use_internal_partition,
          eItemKind cell_kind,Int32ConstArrayView local_ids,Integer nb_node)
{
  // Seul le sous-domain maitre lit les valeurs. Par contre, les autres
  // sous-domaines doivent connaitre la liste des variables et groupes créées.
  // Si une donnée porte le nom 'GROUP_*', on considère qu'il s'agit d'un
  // groupe

  OStringStream created_infos_str;
  created_infos_str() << "<?xml version='1.0' ?>\n";
  created_infos_str() << "<infos>";
  IParallelMng* pm = mesh->parallelMng();
  Int32 sid = pm->commRank();
  Integer nb_cell_kind = mesh->nbItem(cell_kind);
  const char* buf = 0;
  if (sid==0){
    bool reading_node = false;
    bool reading_cell = false;
    while ( !vtk_file.isEnd() && ((buf = vtk_file.getNextLine()) != 0)) {
      debug() << "Read line";
      std::istringstream iline(buf);
      std::string data_str;
      iline >> data_str;
      if (VtkFile::isEqualString(data_str,"CELL_DATA")){
        Integer nb_item =0;
        iline >> ws >> nb_item;
        reading_node = false;
        reading_cell = true;
      }
      else if (VtkFile::isEqualString(data_str,"POINT_DATA")){
        Integer nb_item =0;
        iline >> ws >> nb_item;
        reading_node = true;
        reading_cell = false;
      }
      else{
        if (reading_node || reading_cell){
          std::string type_str;
          std::string s_name_str;
          //String name_str;
          bool is_group = false;
          int nb_component = 0;
          iline >> ws >> s_name_str >> ws >> type_str >> ws >> nb_component;
          debug() << "** ** ** READNAME: name=" << s_name_str << " type=" << type_str;
          String name_str = s_name_str;
          String cstr = name_str.substring(0,6);
          if (cstr=="GROUP_"){
            is_group = true;
            String new_name = name_str.substring(6);
            debug() << "** ** ** GROUP ! name=" << new_name;
            name_str = new_name;
          }
          if (!VtkFile::isEqualString(data_str,"SCALARS")){
            error() << "Expecting 'SCALARS' data type, found=" << data_str;
            return true;
          }
          if (is_group){
            if (!VtkFile::isEqualString(type_str,"int")){
              error() << "Group type must be 'int', found=" << type_str;
              return true;
            }
            // Pour lire LOOKUP_TABLE
            buf = vtk_file.getNextLine();
            if (reading_node){
              created_infos_str() << "<node-group name='" << name_str << "'/>";
              _readNodeGroup(mesh,vtk_file,name_str,nb_node);
            }
            if (reading_cell){
              created_infos_str() << "<cell-group name='" << name_str << "'/>";
              _readItemGroup(mesh,vtk_file,name_str,nb_cell_kind,cell_kind,local_ids);
            }
          }
          else{
            if (!VtkFile::isEqualString(type_str,"float") &&  !VtkFile::isEqualString(type_str,"double")) {
              error() << "Expecting 'float' or 'double' data type, found=" << type_str;
              return true;
            }
            // Pour lire LOOKUP_TABLE
            /*buf = */ vtk_file.getNextLine();
            if (reading_node){
              fatal() << "Unable to read POINT_DATA: feature not implemented";
            }
            if (reading_cell){
              created_infos_str() << "<cell-variable name='" << name_str << "'/>";
              if (cell_kind!=IK_Cell)
                throw IOException("Unable to read face variables: feature not supported");
              _readCellVariable(mesh,vtk_file,name_str,nb_cell_kind);
            }
          }
        }
        else{
          error() << "Expecting value CELL_DATA or POINT_DATA, found='" << data_str << "'";
          return true;
        }
      }
    }
  }
  created_infos_str() << "</infos>";
  if (use_internal_partition){
    ByteUniqueArray bytes;
    if (sid==0){
      String str = created_infos_str.str();
      ByteConstArrayView bv = str.utf8();
      Integer len = bv.size();
      bytes.resize(len+1);
      bytes.copy(bv);
    }
    pm->broadcastMemoryBuffer(bytes,0);
    if (sid!=0){
      String str = String::fromUtf8(bytes);
      info() << "FOUND STR=" << bytes.size() << " " << str;
      ScopedPtrT<IXmlDocumentHolder> doc(IXmlDocumentHolder::loadFromBuffer(bytes,"InternalBuffer",traceMng()));
      XmlNode doc_node = doc->documentNode();
      // Lecture des variables
      {
        XmlNodeList vars = doc_node.documentElement().children("cell-variable");
        for( XmlNode xnode : vars.range() ){
          String name = xnode.attrValue("name");
          info() << "Building variable: " << name;
          VariableCellReal * var = new VariableCellReal(VariableBuildInfo(mesh,name));
          m_variables.add(var);
        }
      }
      // Lecture des groupes de mailles
      {
        XmlNodeList vars = doc_node.documentElement().children("cell-group");
        IItemFamily* cell_family = mesh->itemFamily(cell_kind);
        for( XmlNode xnode : vars.range() ){
          String name = xnode.attrValue("name");
          info() << "Building group: " << name;
          cell_family->createGroup(name);
        }
      }
      // Lecture des groupes de noeuds
      {
        XmlNodeList vars = doc_node.documentElement().children("node-group");
        IItemFamily* node_family = mesh->nodeFamily();
        for( XmlNode xnode : vars.range() ){
          String name = xnode.attrValue("name");
          info() << "Create node group: " << name;
          node_family->createGroup(name);
        }
      }
    }
  }
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkMeshIOService::
_createFaceGroup(IMesh* mesh,const String& name,Int32ConstArrayView faces_lid)
{
  info() << "Building face group '" << name << "'"
         << " size=" << faces_lid.size();

  mesh->faceFamily()->createGroup(name,faces_lid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkMeshIOService::
_readCellVariable(IMesh* mesh,VtkFile& vtk_file,const String& var_name,Integer nb_cell)
{
  //TODO Faire la conversion uniqueId() vers localId() correcte
  info() << "Reading values for variable: " << var_name << " n=" << nb_cell;
  VariableCellReal * var = new VariableCellReal(VariableBuildInfo(mesh,var_name));
  m_variables.add(var);
  RealArrayView values(var->asArray());
  for( Integer i=0; i<nb_cell; ++i ){
    Real v = vtk_file.getReal();
    values[i] = v;
  }
  info() << "Variable build finished: " << vtk_file.isEnd();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkMeshIOService::
_readItemGroup(IMesh* mesh,VtkFile& vtk_file,const String& name,Integer nb_item,
               eItemKind ik,Int32ConstArrayView local_ids)
{
  IItemFamily* item_family = mesh->itemFamily(ik);
  info() << "Reading group info for group: " << name;
  
  Int32UniqueArray ids;
  for( Integer i=0; i<nb_item; ++i ){
    Integer v = vtk_file.getInteger();
    if (v!=0)
      ids.add(local_ids[i]);
  }
  info() << "Building group: " << name << " nb_element=" << ids.size();

  item_family->createGroup(name,ids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkMeshIOService::
_readNodeGroup(IMesh* mesh,VtkFile& vtk_file,const String& name,Integer nb_item)
{
  IItemFamily* item_family = mesh->itemFamily(IK_Node);
  info() << "Lecture infos groupes de noeuds pour le groupe: " << name;
  
  Int32UniqueArray ids;
  for( Integer i=0; i<nb_item; ++i ){
    Integer v = vtk_file.getInteger();
    if (v!=0)
      ids.add(i);
  }
  info() << "Création groupe: " << name << " nb_element=" << ids.size();

  item_family->createGroup(name,ids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VtkLegacyMeshWriter
: public BasicService
, public IMeshWriter
{
 public:
  VtkLegacyMeshWriter(const ServiceBuildInfo& sbi) : BasicService(sbi) {}
 public:
  virtual void build() {}
 public:
  virtual bool writeMeshToFile(IMesh* mesh,const String& file_name);
 private:
  void _writeMeshToFile(IMesh* mesh,const String& file_name,eItemKind cell_kind);
  void _saveGroups(IItemFamily* family,std::ostream& ofile);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SUB_DOMAIN_FACTORY(VtkLegacyMeshWriter,IMeshWriter,VtkLegacyMeshWriter);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ecriture du maillage au Vtk.
 *
 * Pour pouvoir sauver les informations à la fois des mailles et des faces
 * avec les groupes correspondants, on fait deux fichiers. Le premier
 * contient la connectivité et les groupes de mailles, le second la même
 * chose mais pour les faces.
 *
 * Seules les informations de connectivité et les groupes sont sauvés. Les
 * variables ne le sont pas.
 *
 * Le type de DATASET est toujours UNSTRUCTURED_GRID, meme si le
 * maillage est structuré.
 */
bool VtkLegacyMeshWriter::
writeMeshToFile(IMesh* mesh,const String& file_name)
{
  String fname = file_name;
  _writeMeshToFile(mesh,fname,IK_Cell);
  _writeMeshToFile(mesh,fname+"faces.vtk",IK_Face);
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ecrit le maillage au format Vtk.
 *
 * \a cell_kind indique le genre des entités à sauver comme des mailles.
 * Cela peut-être IK_Cell ou IK_Face.
 */
void VtkLegacyMeshWriter::
_writeMeshToFile(IMesh* mesh,const String& file_name,eItemKind cell_kind)
{
  std::ofstream ofile(file_name.localstr());
  ofile.precision(FloatInfo<Real>::maxDigit());
  if (!ofile)
    throw IOException("VtkMeshIOService::writeMeshToFile(): Unable to open file");
  ofile << "# vtk DataFile Version 2.0\n";
  ofile << "Maillage Arcane\n";
  ofile << "ASCII\n";
  ofile << "DATASET UNSTRUCTURED_GRID\n";
  UniqueArray<Integer> nodes_local_id_to_current(mesh->itemFamily(IK_Node)->maxLocalId());
  nodes_local_id_to_current.fill(NULL_ITEM_ID);

  Integer nb_node = mesh->nbNode();
  IItemFamily* cell_kind_family = mesh->itemFamily(cell_kind);
  Integer nb_cell_kind = cell_kind_family->nbItem();

  // Sauve les noeuds
  {
    ofile << "POINTS " << nb_node << " double\n";
    VariableNodeReal3& coords(mesh->toPrimaryMesh()->nodesCoordinates());
    Integer node_index = 0;
    ENUMERATE_NODE(inode,mesh->allNodes()){
      const Node& node = *inode;
      nodes_local_id_to_current[node.localId()] = node_index;
      Real3 xyz = coords[inode];
      ofile << xyz.x << ' ' << xyz.y << ' ' << xyz.z << '\n';
      ++node_index;
    }
  }

  // Sauve les mailles ou faces
  {
    Integer nb_node_cell_kind = nb_cell_kind;
    ENUMERATE_ITEMWITHNODES(iitem,cell_kind_family->allItems()){
      nb_node_cell_kind += (*iitem).nbNode();
    }
    ofile << "CELLS " << nb_cell_kind << ' ' << nb_node_cell_kind << "\n";
    ENUMERATE_ITEMWITHNODES(iitem,cell_kind_family->allItems()){
      const ItemWithNodes& item = *iitem;
      Integer item_nb_node = item.nbNode();
      ofile << item_nb_node;
      for( NodeEnumerator inode(item.nodes()); inode(); ++inode ){
        ofile << ' ' << nodes_local_id_to_current[inode->localId()];
      }
      ofile << '\n';
    }
    // Le type doit être coherent avec celui de vtkCellType.h
    ofile << "CELL_TYPES " << nb_cell_kind << "\n";
    ENUMERATE_ITEMWITHNODES(iitem,cell_kind_family->allItems()){
      int type = 0; // Correspond à VTK_EMPTY_CELL
      int arcane_type = (*iitem).type();
      switch(arcane_type){
      case IT_NullType: type = VTK_EMPTY_CELL; break;
      case IT_Vertex: type = VTK_VERTEX; break;
      case IT_Line2: type = VTK_LINE; break;
      case IT_Line3: type = VTK_QUADRATIC_EDGE; break;
      case IT_Triangle3: type = VTK_TRIANGLE; break;
      case IT_Triangle6: type = VTK_QUADRATIC_TRIANGLE; break;
      case IT_Quad4: type = VTK_QUAD; break;
      case IT_Quad8: type = VTK_QUADRATIC_QUAD; break;
      case IT_Pentagon5: type = VTK_POLYGON; break; // VTK_POLYGON (a tester...)
      case IT_Hexagon6: type = VTK_POLYGON; break; // VTK_POLYGON (a tester ...)
      case IT_Tetraedron4: type = VTK_TETRA; break;
      case IT_Tetraedron10: type = VTK_QUADRATIC_TETRA; break;
      case IT_Pyramid5: type = VTK_PYRAMID; break;
      case IT_Pentaedron6: type = VTK_WEDGE; break;
      case IT_Hexaedron8: type = VTK_HEXAHEDRON; break;
      case IT_Hexaedron20: type = VTK_QUADRATIC_HEXAHEDRON; break;
      case IT_Heptaedron10: type = VTK_PENTAGONAL_PRISM; break;
      case IT_Octaedron12: type = VTK_HEXAGONAL_PRISM; break;
      default:
        ARCANE_FATAL("Unsuported item type for VtkWriter type={0}",arcane_type);
      }
      ofile << type << '\n';
    }
  }

  // Si on est dans le maillage des mailles, sauve les groupes de noeuds.
  if (cell_kind==IK_Cell){
    ofile << "POINT_DATA " << nb_node << "\n";
    _saveGroups(mesh->itemFamily(IK_Node),ofile);
  }

  // Sauve les groupes de mailles
  ofile << "CELL_DATA " << nb_cell_kind << "\n";
  _saveGroups(mesh->itemFamily(cell_kind),ofile);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkLegacyMeshWriter::
_saveGroups(IItemFamily* family,std::ostream& ofile)
{
  info() << "Saving groups for family name=" << family->name();
  UniqueArray<char> in_group_list(family->maxLocalId());
  for( ItemGroupCollection::Enumerator igroup(family->groups()); ++igroup; ){
    ItemGroup group = *igroup;
    // Inutile de sauver le groupe de toutes les entités
    if (group==family->allItems())
      continue;
    //HACK: a supprimer
    if (group.name()=="OuterFaces")
      continue;
    ofile << "SCALARS GROUP_" << group.name() << " int 1\n";
    ofile << "LOOKUP_TABLE default\n";
    in_group_list.fill('0');
    ENUMERATE_ITEM(iitem,group){
      in_group_list[(*iitem).localId()] = '1';
    }
    ENUMERATE_ITEM(iitem,family->allItems()){
      ofile << in_group_list[(*iitem).localId()] << '\n';
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VtkMeshReader
: public AbstractService
, public IMeshReader
{
 public:

  explicit VtkMeshReader(const ServiceBuildInfo& sbi)
  : AbstractService(sbi){}

 public:

  bool allowExtension(const String& str) override { return str=="vtk"; }
  eReturnType readMeshFromFile(IPrimaryMesh* mesh,const XmlNode& mesh_node,const String& file_name,
                               const String& dir_name,bool use_internal_partition) override

  {
    ARCANE_UNUSED(mesh_node);
    VtkMeshIOService vtk_service(traceMng());
    bool ret = vtk_service.readMesh(mesh,file_name,dir_name,use_internal_partition);
    if (ret)
      return RTError;

    return RTOk;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SUB_DOMAIN_FACTORY(VtkMeshReader,IMeshReader,VtkMeshIO);

ARCANE_REGISTER_SERVICE(VtkMeshReader,
                        ServiceProperty("VtkLegacyMeshReader",ST_SubDomain),
                        ARCANE_SERVICE_INTERFACE(IMeshReader));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VtkLegacyCaseMeshReader
: public AbstractService
, public ICaseMeshReader
{
 public:
  class Builder
  : public IMeshBuilder
  {
   public:
    explicit Builder(ITraceMng* tm,const CaseMeshReaderReadInfo& read_info)
    : m_trace_mng(tm), m_read_info(read_info) {}
   public:
    void fillMeshBuildInfo(MeshBuildInfo& build_info) override
    {
      ARCANE_UNUSED(build_info);
    }
    void allocateMeshItems(IPrimaryMesh* pm) override
    {
      VtkMeshIOService vtk_service(m_trace_mng);
      String fname = m_read_info.fileName();
      m_trace_mng->info() << "VtkLegacy Reader (ICaseMeshReader) file_name=" << fname;
      bool ret = vtk_service.readMesh(pm,fname,m_read_info.directoryName(),m_read_info.isParallelRead());
      if (ret)
        ARCANE_FATAL("Can not read VTK File");
    }
   private:
    ITraceMng* m_trace_mng;
    CaseMeshReaderReadInfo m_read_info;
  };
 public:

  explicit VtkLegacyCaseMeshReader(const ServiceBuildInfo& sbi)
  : AbstractService(sbi){}

 public:

  Ref<IMeshBuilder> createBuilder(const CaseMeshReaderReadInfo& read_info) const override
  {
    IMeshBuilder* builder = nullptr;
    if (read_info.format()=="vtk")
      builder = new Builder(traceMng(),read_info);
    return makeRef(builder);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(VtkLegacyCaseMeshReader,
                        ServiceProperty("VtkLegacyCaseMeshReader",ST_SubDomain),
                        ARCANE_SERVICE_INTERFACE(ICaseMeshReader));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
