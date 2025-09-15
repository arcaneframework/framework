// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VtkMeshIOService.cc                                         (C) 2000-2025 */
/*                                                                           */
/* Lecture/Ecriture d'un maillage au format Vtk historique (legacy).         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Collection.h"
#include "arcane/utils/Enumerator.h"
#include "arcane/utils/HashTableMap.h"
#include "arcane/utils/IOException.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/Iostream.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/StdHeader.h"
#include "arcane/utils/String.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/Real3.h"

#include "arcane/core/BasicService.h"
#include "arcane/core/FactoryService.h"
#include "arcane/core/ICaseMeshReader.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/IPrimaryMesh.h"
#include "arcane/core/IMeshBuilder.h"
#include "arcane/core/IMeshReader.h"
#include "arcane/core/IMeshUtilities.h"
#include "arcane/core/IMeshWriter.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/IVariableAccessor.h"
#include "arcane/core/IXmlDocumentHolder.h"
#include "arcane/core/Item.h"
#include "arcane/core/ItemEnumerator.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/core/VariableTypes.h"
#include "arcane/core/XmlNode.h"
#include "arcane/core/XmlNodeList.h"
#include "arcane/core/UnstructuredMeshAllocateBuildInfo.h"

#include "arcane/core/internal/IVariableMngInternal.h"
#include "arcane/core/internal/VtkCellTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VtkFile;
using namespace Arcane::VtkUtils;

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
 * # vtk DataFile Version X.X
 * Où X.X est la version du fichier VTK (support des fichiers VTK <= 4.2).
 *
 * Il est possible de spécifier un ensemble de variables dans le fichier.
 * Dans ce cas, leurs valeurs sont lues en même temps que le maillage
 * et servent à initialiser les variables. Actuellement, seules les valeurs
 * aux mailles sont supportées.
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
  : TraceAccessor(tm)
  {}

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

  bool readMesh(IPrimaryMesh* mesh, const String& file_name, const String& dir_name, bool use_internal_partition);

 private:

  bool _readStructuredGrid(IPrimaryMesh* mesh, VtkFile&, bool use_internal_partition);
  bool _readUnstructuredGrid(IPrimaryMesh* mesh, VtkFile& vtk_file, bool use_internal_partition);
  void _readCellVariable(IMesh* mesh, VtkFile& vtk_file, const String& name_str, Integer nb_cell);
  void _readItemGroup(IMesh* mesh, VtkFile& vtk_file, const String& name_str, Integer nb_item,
                      eItemKind ik, ConstArrayView<Int32> local_id);
  void _readNodeGroup(IMesh* mesh, VtkFile& vtk_file, const String& name, Integer nb_item);
  void _createFaceGroup(IMesh* mesh, const String& name, Int32ConstArrayView faces_lid);
  bool _readData(IMesh* mesh, VtkFile& vtk_file, bool use_internal_partition, eItemKind cell_kind,
                 Int32ConstArrayView local_id, Integer nb_node);
  void _readNodesUnstructuredGrid(IMesh* mesh, VtkFile& vtk_file, Array<Real3>& node_coords);
  void _readCellsUnstructuredGrid(IMesh* mesh, VtkFile& vtk_file,
                                  Array<Integer>& cells_nb_node,
                                  Array<ItemTypeId>& cells_type,
                                  Array<Int64>& cells_connectivity);
  void _readFacesMesh(IMesh* mesh, const String& file_name,
                      const String& dir_name, bool use_internal_partition);
  bool _readMetadata(IMesh* mesh, VtkFile& vtk_file);

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VtkFile
{
 public:

  static const int BUFSIZE = 10000;

 public:

  explicit VtkFile(std::istream* stream)
  : m_stream(stream)
  , m_is_init(false)
  , m_need_reread_current_line(false)
  , m_is_eof(false)
  , m_is_binary_file(false)
  , m_buf{}
  {}

  const char* getCurrentLine();
  bool isEmptyNextLine();
  const char* getNextLine();

  void checkString(const String& current_value, const String& expected_value);
  void checkString(const String& current_value,
                   const String& expected_value1,
                   const String& expected_value2);

  static bool isEqualString(const String& current_value, const String& expected_value);

  void reReadSameLine() { m_need_reread_current_line = true; }

  bool isEof() { return m_is_eof; }

  template <class T>
  void getBinary(T& type);
  float getFloat();
  double getDouble();
  int getInt();

  void setIsBinaryFile(bool new_val) { m_is_binary_file = new_val; }

 private:

  //! Le stream.
  std::istream* m_stream = nullptr;

  //! Y'a-t-il eu au moins une ligne lue.
  bool m_is_init;

  //! Doit-on relire la même ligne.
  bool m_need_reread_current_line;

  //! Est-on à la fin du fichier.
  bool m_is_eof;

  //! Est-ce un fichier contenant des données en binaire.
  bool m_is_binary_file;

  //! Le buffer contenant la ligne lue.
  char m_buf[BUFSIZE];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Permet de retourner la ligne présente dans le buffer.
 *
 * \return le buffer contenant la dernière ligne lue
 */
const char* VtkFile::
getCurrentLine()
{
  if (!m_is_init)
    getNextLine();
  return m_buf;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Permet de voir si la prochaine ligne est vide.
 *        
 * A la fin de cette méthode, le buffer contiendra la prochaine ligne
 * non vide. Le booléen m_need_reread_current_line permettera de demander à getNextLine
 * de renvoyer cette ligne qui n'a pas été lue.
 * 
 * \return true s'il y a une ligne vide, false sinon
 */
bool VtkFile::
isEmptyNextLine()
{
  m_is_init = true;

  // On veut que getNextLine lise une nouvelle ligne.
  // (on met à false dans le cas où cette méthode serai
  // appelée plusieurs fois à la suite).
  m_need_reread_current_line = false;

  // Si l'on est arrivé à la fin du fichier lors du précédent appel de cette méthode ou
  // de getNextLine, on envoie une erreur.
  if (m_is_eof) {
    throw IOException("VtkFile::isEmptyNextLine()", "Unexpected EndOfFile");
  }

  if (m_stream->good()) {
    // Le getline s'arrete (par défaut) au char '\n' et ne l'inclus pas dans le buf
    // mais le remplace par '\0'.
    m_stream->getline(m_buf, sizeof(m_buf) - 1);

    // Si on arrive au bout du fichier, on return true (pour dire oui, il y a une ligne vide,
    // à l'appelant de gérer ça).
    if (m_stream->eof()) {
      m_is_eof = true;
      return true;
    }

    // Sous Windows, une ligne vide commence par \r.
    // getline remplace \n par \0, que ce soit sous Windows ou Linux.
    if (m_buf[0] == '\r' || m_buf[0] == '\0') {
      getNextLine();

      // On demande à ce que le prochain appel à getNextLine renvoie la ligne
      // qui vient tout juste d'être bufferisée.
      m_need_reread_current_line = true;
      return true;
    }
    else {
      bool is_comment = true;

      // On retire le commentaire, s'il y en a un, en remplaçant '#' par '\0'.
      for (int i = 0; i < BUFSIZE && m_buf[i] != '\0'; ++i) {
        if (!isspace(m_buf[i]) && m_buf[i] != '#' && is_comment) {
          is_comment = false;
        }
        if (m_buf[i] == '#') {
          m_buf[i] = '\0';
          break;
        }
      }

      // Si ce n'est pas un commentaire, on supprime juste le '\r' final (si windows).
      if (!is_comment) {
        // Supprime le '\r' final
        for (int i = 0; i < BUFSIZE && m_buf[i] != '\0'; ++i) {
          if (m_buf[i] == '\r') {
            m_buf[i] = '\0';
            break;
          }
        }
      }

      // Si c'était un commentaire, on recherche la prochaine ligne "valide"
      // en appelant getNextLine.
      else {
        getNextLine();
      }
    }
    m_need_reread_current_line = true;
    return false;
  }
  throw IOException("VtkFile::isEmptyNextLine()", "Not Good");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Permet de récupérer la prochaine ligne du fichier.
 *
 * \return le buffer contenant la dernière ligne lue
 */
const char* VtkFile::
getNextLine()
{
  m_is_init = true;

  // On return le buffer actuel, si celui-ci n'a pas été utilisé.
  if (m_need_reread_current_line) {
    m_need_reread_current_line = false;
    return getCurrentLine();
  }

  // Si l'on est arrivé à la fin du fichier lors du précédent appel de cette méthode ou
  // de isEmptyNextLine, on envoie une erreur.
  if (m_is_eof) {
    throw IOException("VtkFile::isEmptyNextLine()", "Unexpected EndOfFile");
  }

  while (m_stream->good()) {
    // Le getline s'arrete (par défaut) au char '\n' et ne l'inclus pas dans le buf mais le remplace par '\0'.
    m_stream->getline(m_buf, sizeof(m_buf) - 1);

    // Si on arrive au bout du fichier, on return le buffer avec \0 au début (c'est à l'appelant d'appeler
    // isEof() pour savoir si le fichier est fini ou non).
    if (m_stream->eof()) {
      m_is_eof = true;
      m_buf[0] = '\0';
      return m_buf;
    }

    bool is_comment = true;

    // Sous Windows, une ligne vide commence par \r.
    // getline remplace \n par \0, que ce soit sous Windows ou Linux.
    if (m_buf[0] == '\0' || m_buf[0] == '\r')
      continue;

    // On retire le commentaire, s'il y en a un, en remplaçant '#' par '\0'.
    for (int i = 0; i < BUFSIZE && m_buf[i] != '\0'; ++i) {
      if (!isspace(m_buf[i]) && m_buf[i] != '#' && is_comment) {
        is_comment = false;
      }
      if (m_buf[i] == '#') {
        m_buf[i] = '\0';
        break;
      }
    }

    // Si ce n'est pas un commentaire, on supprime juste le '\r' final (si windows).
    if (!is_comment) {
      // Supprime le '\r' final
      for (int i = 0; i < BUFSIZE && m_buf[i] != '\0'; ++i) {
        if (m_buf[i] == '\r') {
          m_buf[i] = '\0';
          break;
        }
      }
      return m_buf;
    }
  }
  throw IOException("VtkFile::getNextLine()", "Not good");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Permet de récupérer le float qui suit.
 *
 * \return le float récupéré
 */
float VtkFile::
getFloat()
{
  float v = 0.;
  if (m_is_binary_file) {
    getBinary(v);
    return v;
  }
  (*m_stream) >> ws >> v;

  if (m_stream->good())
    return v;

  throw IOException("VtkFile::getFloat()", "Bad float");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Permet de récupérer le double qui suit.
 *
 * \return le double récupéré
 */
double VtkFile::
getDouble()
{
  double v = 0.;
  if (m_is_binary_file) {
    getBinary(v);
    return v;
  }
  (*m_stream) >> ws >> v;

  if (m_stream->good())
    return v;

  throw IOException("VtkFile::getDouble()", "Bad double");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Permet de récupérer le int qui suit.
 *
 * \return le int récupéré
 */
int VtkFile::
getInt()
{
  int v = 0;
  if (m_is_binary_file) {
    getBinary(v);
    return v;
  }
  (*m_stream) >> ws >> v;

  if (m_stream->good())
    return v;

  throw IOException("VtkFile::getInt()", "Bad int");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Permet de récupérer le nombre binaire qui suit.
 *
 * \param value La référence vers la variable à remplir (le type de value nous renseigne sur le nombre d'octet à lire).
 */
template <class T>
void VtkFile::
getBinary(T& value)
{
  constexpr size_t sizeofT = sizeof(T);

  // Le fichier VTK est en big endian et les CPU actuels sont en little endian.
  Byte big_endian[sizeofT];
  Byte little_endian[sizeofT];

  // On lit les 'sizeofT' prochains octets que l'on met dans big_endian.
  m_stream->read((char*)big_endian, sizeofT);

  // On transforme le big_endian en little_endian.
  for (size_t i = 0; i < sizeofT; i++) {
    little_endian[sizeofT - 1 - i] = big_endian[i];
  }

  // On 'cast' la liste d'octet en type 'T'.
  T* conv = new (little_endian) T;
  value = *conv;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Permet de vérifier si expected_value == current_value.
 *
 * Permet de vérifier si expected_value correspond à current_value. 
 * Une exception est envoyée sinon.
 *
 * \param current_value la valeur référence
 * \param expected_value la valeur à comparer
 */
void VtkFile::
checkString(const String& current_value, const String& expected_value)
{
  String current_value_low = current_value.lower();
  String expected_value_low = expected_value.lower();

  if (current_value_low != expected_value_low) {
    String s = "Expecting chain '" + expected_value + "', found '" + current_value + "'";
    throw IOException("VtkFile::checkString()", s);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Permet de vérifier si expected_value1 ou expected_value2 == current_value.
 *
 * Permet de vérifier si expected_value1 ou expected_value2 correspond à current_value. 
 * Une exception est envoyée sinon.
 *
 * \param current_value la valeur référence
 * \param expected_value1 la première valeur à comparer
 * \param expected_value2 la deuxième valeur à comparer
 */
void VtkFile::
checkString(const String& current_value, const String& expected_value1, const String& expected_value2)
{
  String current_value_low = current_value.lower();
  String expected_value1_low = expected_value1.lower();
  String expected_value2_low = expected_value2.lower();

  if (current_value_low != expected_value1_low && current_value_low != expected_value2_low) {
    String s = "Expecting chain '" + expected_value1 + "' or '" + expected_value2 + "', found '" + current_value + "'";
    throw IOException("VtkFile::checkString()", s);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Permet de vérifier si expected_value == current_value.
 *
 * Permet de vérifier si expected_value correspond à current_value. 
 * 
 * \param current_value la valeur référence
 * \param expected_value la valeur à comparer
 * \return true si les valeurs sont égales, false sinon
 */
bool VtkFile::
isEqualString(const String& current_value, const String& expected_value)
{
  String current_value_low = current_value.lower();
  String expected_value_low = expected_value.lower();
  return (current_value_low == expected_value_low);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Permet de débuter la lecture d'un fichier vtk.
 *
 * \param mesh Le maillage à remplir
 * \param file_name Le nom du fichier vtk (avec l'extension)
 * \param dir_name Le chemin du fichier
 * \param use_internal_partition Doit-on utiliser le partitionneur interne ou non
 * \return false si tout s'est bien passé, true sinon
 */
bool VtkMeshIOService::
readMesh(IPrimaryMesh* mesh, const String& file_name, const String& dir_name, bool use_internal_partition)
{
  std::ifstream ifile(file_name.localstr(), std::ifstream::binary);

  if (!ifile) {
    error() << "Unable to read file '" << file_name << "'";
    return true;
  }

  debug() << "Fichier ouvert : " << file_name.localstr();

  VtkFile vtk_file(&ifile);
  const char* buf = 0;

  // Lecture de la description
  // Lecture title.
  String title = vtk_file.getNextLine();

  info() << "Titre du fichier VTK : " << title.localstr();

  // Lecture format.
  String format = vtk_file.getNextLine();

  debug() << "Format du fichier VTK : " << format.localstr();

  if (VtkFile::isEqualString(format, "BINARY")) {
    vtk_file.setIsBinaryFile(true);
  }

  eMeshType mesh_type = VTK_MT_Unknown;

  // Lecture du type de maillage
  // TODO: en parallèle, avec use_internal_partition vrai, seul le processeur 0
  // lit les données. Dans ce cas, inutile que les autres ouvrent le fichier.
  {
    buf = vtk_file.getNextLine();

    std::istringstream mesh_type_line(buf);
    std::string dataset_str;
    std::string mesh_type_str;

    mesh_type_line >> ws >> dataset_str >> ws >> mesh_type_str;

    vtk_file.checkString(dataset_str, "DATASET");

    if (VtkFile::isEqualString(mesh_type_str, "STRUCTURED_GRID")) {
      mesh_type = VTK_MT_StructuredGrid;
    }

    if (VtkFile::isEqualString(mesh_type_str, "UNSTRUCTURED_GRID")) {
      mesh_type = VTK_MT_UnstructuredGrid;
    }

    if (mesh_type == VTK_MT_Unknown) {
      error() << "Support exists only for 'STRUCTURED_GRID' and 'UNSTRUCTURED_GRID' formats (format=" << mesh_type_str << "')";
      return true;
    }
  }
  debug() << "Lecture en-tête OK";

  bool ret = true;
  switch (mesh_type) {
  case VTK_MT_StructuredGrid:
    ret = _readStructuredGrid(mesh, vtk_file, use_internal_partition);
    break;

  case VTK_MT_UnstructuredGrid:
    ret = _readUnstructuredGrid(mesh, vtk_file, use_internal_partition);
    debug() << "Lecture _readUnstructuredGrid OK";
    if (!ret) {
      // Tente de lire le fichier des faces s'il existe
      _readFacesMesh(mesh, file_name + "faces.vtk", dir_name, use_internal_partition);
      debug() << "Lecture _readFacesMesh OK";
    }
    break;

  case VTK_MT_Unknown:
    break;
  }
  /*while ( (buf=vtk_file.getNextLine()) != 0 ){
    info() << " STR " << buf;
    }*/

  ifile.close();
  return ret;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Permet de lire un fichier vtk contenant une STRUCTURED_GRID.
 *
 * \param mesh Le maillage à remplir
 * \param vtk_file Référence vers un objet VtkFile
 * \param use_internal_partition Doit-on utiliser le partitionneur interne ou non
 * \return false si tout s'est bien passé, true sinon
 */
bool VtkMeshIOService::
_readStructuredGrid(IPrimaryMesh* mesh, VtkFile& vtk_file, bool use_internal_partition)
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
    iline >> ws >> dimension_str >> ws >> nb_node_x >> ws >> nb_node_y >> ws >> nb_node_z;

    if (!iline) {
      error() << "Syntax error while reading grid dimensions";
      return true;
    }

    vtk_file.checkString(dimension_str, "DIMENSIONS");
    if (nb_node_x <= 1 || nb_node_y <= 1 || nb_node_z <= 1) {
      error() << "Invalid dimensions: x=" << nb_node_x << " y=" << nb_node_y << " z=" << nb_node_z;
      return true;
    }
  }
  info() << " Infos: " << nb_node_x << " " << nb_node_y << " " << nb_node_z;
  Integer nb_node = nb_node_x * nb_node_y * nb_node_z;

  // Lecture du nombre de points: POINTS nb float
  std::string float_str;
  {
    buf = vtk_file.getNextLine();
    std::istringstream iline(buf);
    std::string points_str;
    Integer nb_node_read = 0;
    iline >> ws >> points_str >> ws >> nb_node_read >> ws >> float_str;
    if (!iline) {
      error() << "Syntax error while reading grid dimensions";
      return true;
    }
    vtk_file.checkString(points_str, "POINTS");
    if (nb_node_read != nb_node) {
      error() << "Number of invalid nodes: expected=" << nb_node << " found=" << nb_node_read;
      return true;
    }
  }

  Int32 rank = mesh->parallelMng()->commRank();

  Integer nb_cell_x = nb_node_x - 1;
  Integer nb_cell_y = nb_node_y - 1;
  Integer nb_cell_z = nb_node_z - 1;

  if (use_internal_partition && rank != 0) {
    nb_node_x = 0;
    nb_node_y = 0;
    nb_node_z = 0;
    nb_cell_x = 0;
    nb_cell_y = 0;
    nb_cell_z = 0;
  }

  const Integer nb_node_yz = nb_node_y * nb_node_z;
  const Integer nb_node_xy = nb_node_x * nb_node_y;

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
      for (Integer x = 0; x < nb_node_x; ++x) {
        for (Integer z = 0; z < nb_node_z; ++z) {
          for (Integer y = 0; y < nb_node_y; ++y) {

            Integer node_unique_id = y + (z)*nb_node_y + x * nb_node_y * nb_node_z;

            nodes_unique_id[node_local_id] = node_unique_id;

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
    UniqueArray<Int64> cells_infos(nb_cell * 10);

    {
      Integer cell_local_id = 0;
      Integer cells_infos_index = 0;

      // Normalement ne doit pas arriver car les valeurs de nb_node_x et
      // nb_node_y sont testées lors de la lecture.
      if (nb_node_xy == 0)
        ARCANE_FATAL("Null value for nb_node_xy");

      //Integer index = 0;
      for (Integer z = 0; z < nb_cell_z; ++z) {
        for (Integer y = 0; y < nb_cell_y; ++y) {
          for (Integer x = 0; x < nb_cell_x; ++x) {
            Integer current_cell_nb_node = 8;

            //Integer cell_unique_id = y + (z)*nb_cell_y + x*nb_cell_y*nb_cell_z;
            Int64 cell_unique_id = x + y * nb_cell_x + z * nb_cell_x * nb_cell_y;

            cells_infos[cells_infos_index] = IT_Hexaedron8;
            ++cells_infos_index;

            cells_infos[cells_infos_index] = cell_unique_id;
            ++cells_infos_index;

            //Integer base_id = y + z*nb_node_y + x*nb_node_yz;
            Integer base_id = x + y * nb_node_x + z * nb_node_xy;
            cells_infos[cells_infos_index + 0] = nodes_unique_id[base_id];
            cells_infos[cells_infos_index + 1] = nodes_unique_id[base_id + 1];
            cells_infos[cells_infos_index + 2] = nodes_unique_id[base_id + nb_node_x + 1];
            cells_infos[cells_infos_index + 3] = nodes_unique_id[base_id + nb_node_x + 0];
            cells_infos[cells_infos_index + 4] = nodes_unique_id[base_id + nb_node_xy];
            cells_infos[cells_infos_index + 5] = nodes_unique_id[base_id + nb_node_xy + 1];
            cells_infos[cells_infos_index + 6] = nodes_unique_id[base_id + nb_node_xy + nb_node_x + 1];
            cells_infos[cells_infos_index + 7] = nodes_unique_id[base_id + nb_node_xy + nb_node_x + 0];
            cells_infos_index += current_cell_nb_node;
            cells_local_id[cell_local_id] = cell_local_id;
            ++cell_local_id;
          }
        }
      }
    }

    mesh->setDimension(3);
    mesh->allocateCells(nb_cell, cells_infos, false);
    mesh->endAllocate();

    // Positionne les coordonnées
    {
      UniqueArray<Real3> coords(nb_node);

      if (vtk_file.isEqualString(float_str, "int")) {
        for (Integer z = 0; z < nb_node_z; ++z) {
          for (Integer y = 0; y < nb_node_y; ++y) {
            for (Integer x = 0; x < nb_node_x; ++x) {
              Real nx = vtk_file.getInt();
              Real ny = vtk_file.getInt();
              Real nz = vtk_file.getInt();
              Integer node_unique_id = x + y * nb_node_x + z * nb_node_xy;
              coords[node_unique_id] = Real3(nx, ny, nz);
            }
          }
        }
      }
      else if (vtk_file.isEqualString(float_str, "float")) {
        for (Integer z = 0; z < nb_node_z; ++z) {
          for (Integer y = 0; y < nb_node_y; ++y) {
            for (Integer x = 0; x < nb_node_x; ++x) {
              Real nx = vtk_file.getFloat();
              Real ny = vtk_file.getFloat();
              Real nz = vtk_file.getFloat();
              Integer node_unique_id = x + y * nb_node_x + z * nb_node_xy;
              coords[node_unique_id] = Real3(nx, ny, nz);
            }
          }
        }
      }
      else if (vtk_file.isEqualString(float_str, "double")) {
        for (Integer z = 0; z < nb_node_z; ++z) {
          for (Integer y = 0; y < nb_node_y; ++y) {
            for (Integer x = 0; x < nb_node_x; ++x) {
              Real nx = vtk_file.getDouble();
              Real ny = vtk_file.getDouble();
              Real nz = vtk_file.getDouble();
              Integer node_unique_id = x + y * nb_node_x + z * nb_node_xy;
              coords[node_unique_id] = Real3(nx, ny, nz);
            }
          }
        }
      }
      else {
        throw IOException("_readStructuredGrid", "Invalid type name");
      }

      VariableNodeReal3& nodes_coord_var(mesh->nodesCoordinates());
      ENUMERATE_NODE (inode, mesh->allNodes()) {
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

    ENUMERATE_FACE (iface, mesh->allFaces()) {
      Face face = *iface;
      Integer face_local_id = face.localId();
      bool is_xmin = true;
      bool is_xmax = true;
      bool is_ymin = true;
      bool is_ymax = true;
      bool is_zmin = true;
      bool is_zmax = true;
      for (Node node : face.nodes() ) {
        Int64 node_unique_id = node.uniqueId().asInt64();
        Int64 node_z = node_unique_id / nb_node_xy;
        Int64 node_y = (node_unique_id - node_z * nb_node_xy) / nb_node_x;
        Int64 node_x = node_unique_id - node_z * nb_node_xy - node_y * nb_node_x;
        if (node_x != 0)
          is_xmin = false;
        if (node_x != (nb_node_x - 1))
          is_xmax = false;
        if (node_y != 0)
          is_ymin = false;
        if (node_y != (nb_node_y - 1))
          is_ymax = false;
        if (node_z != 0)
          is_zmin = false;
        if (node_z != (nb_node_z - 1))
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
    _createFaceGroup(mesh, "XMIN", xmin_surface_lid);
    _createFaceGroup(mesh, "XMAX", xmax_surface_lid);
    _createFaceGroup(mesh, "YMIN", ymin_surface_lid);
    _createFaceGroup(mesh, "YMAX", ymax_surface_lid);
    _createFaceGroup(mesh, "ZMIN", zmin_surface_lid);
    _createFaceGroup(mesh, "ZMAX", zmax_surface_lid);
  }

  _readMetadata(mesh, vtk_file);

  // Maintenant, regarde s'il existe des données associées aux fichier
  bool r = _readData(mesh, vtk_file, use_internal_partition, IK_Cell, cells_local_id, nb_node);
  if (r)
    return r;

  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Lecture des noeuds et de leur coordonnées.
 *
 * \param mesh Le maillage à remplir
 * \param vtk_file Référence vers un objet VtkFile
 * \param node_coords L'array à remplir de coordonnées de nodes
 */
void VtkMeshIOService::
_readNodesUnstructuredGrid(IMesh* mesh, VtkFile& vtk_file, Array<Real3>& node_coords)
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
    throw IOException(func_name, "Syntax error while reading number of nodes");

  vtk_file.checkString(points_str, "POINTS");

  if (nb_node < 0)
    throw IOException(A_FUNCINFO, String::format("Invalid number of nodes: n={0}", nb_node));

  info() << "VTK file : number of nodes = " << nb_node;

  // Lecture les coordonnées
  node_coords.resize(nb_node);
  {
    if (vtk_file.isEqualString(data_type_str, "int")) {
      for (Integer i = 0; i < nb_node; ++i) {
        Real nx = vtk_file.getInt();
        Real ny = vtk_file.getInt();
        Real nz = vtk_file.getInt();
        node_coords[i] = Real3(nx, ny, nz);
      }
    }
    else if (vtk_file.isEqualString(data_type_str, "float")) {
      for (Integer i = 0; i < nb_node; ++i) {
        Real nx = vtk_file.getFloat();
        Real ny = vtk_file.getFloat();
        Real nz = vtk_file.getFloat();
        node_coords[i] = Real3(nx, ny, nz);
      }
    }
    else if (vtk_file.isEqualString(data_type_str, "double")) {
      for (Integer i = 0; i < nb_node; ++i) {
        Real nx = vtk_file.getDouble();
        Real ny = vtk_file.getDouble();
        Real nz = vtk_file.getDouble();
        node_coords[i] = Real3(nx, ny, nz);
      }
    }
    else {
      throw IOException(func_name, "Invalid type name");
    }
  }
  _readMetadata(mesh, vtk_file);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Lecture des mailles et de leur connectivité.
 *
 * En retour, remplit \a cells_nb_node, \a cells_type et \a cells_connectivity.
 * 
 * \param mesh Le maillage à remplir
 * \param vtk_file Référence vers un objet VtkFile
 * \param cells_nb_node Nombre de nodes de chaque cell
 * \param cells_type Type de chaque cell
 * \param cells_connectivity Connectivités entre les cells
 */
void VtkMeshIOService::
_readCellsUnstructuredGrid(IMesh* mesh, VtkFile& vtk_file,
                           Array<Integer>& cells_nb_node,
                           Array<ItemTypeId>& cells_type,
                           Array<Int64>& cells_connectivity)
{
  ARCANE_UNUSED(mesh);

  const char* func_name = "VtkMeshIOService::_readCellsUnstructuredGrid()";
  const char* buf = vtk_file.getNextLine();

  //String buftest = vtk_file.getCurrentLine(); // DEBUG
  //pinfo() << "Ligne lu : " << buftest.localstr();

  std::istringstream iline(buf);
  std::string cells_str;
  Integer nb_cell = 0;
  Integer nb_cell_node = 0;

  iline >> ws >> cells_str >> ws >> nb_cell >> ws >> nb_cell_node;

  if (!iline)
    throw IOException(func_name, "Syntax error while reading cells");

  vtk_file.checkString(cells_str, "CELLS");

  info() << "VTK file : number of cells = " << nb_cell;

  if (nb_cell < 0 || nb_cell_node < 0) {
    throw IOException(A_FUNCINFO,
                      String::format("Invalid dimensions: nb_cell={0} nb_cell_node={1}",
                                     nb_cell, nb_cell_node));
  }

  cells_nb_node.resize(nb_cell);
  cells_type.resize(nb_cell);
  cells_connectivity.resize(nb_cell_node);

  {
    Integer connectivity_index = 0;
    for (Integer i = 0; i < nb_cell; ++i) {
      Integer n = vtk_file.getInt();
      cells_nb_node[i] = n;
      for (Integer j = 0; j < n; ++j) {
        Integer id = vtk_file.getInt();
        cells_connectivity[connectivity_index] = id;
        ++connectivity_index;
      }
    }
  }

  _readMetadata(mesh, vtk_file);

  // Lecture du type des mailles
  {
    buf = vtk_file.getNextLine();
    std::istringstream iline(buf);
    std::string cell_types_str;
    Integer nb_cell_type;
    iline >> ws >> cell_types_str >> ws >> nb_cell_type;

    if (!iline) {
      throw IOException(func_name, "Syntax error while reading cell types");
    }

    vtk_file.checkString(cell_types_str, "CELL_TYPES");
    if (nb_cell_type != nb_cell) {
      ARCANE_THROW(IOException, "Inconsistency in number of CELL_TYPES: v={0} nb_cell={1}",
                   nb_cell_type, nb_cell);
    }
  }

  for (Integer i = 0; i < nb_cell; ++i) {
    Integer vtk_ct = vtk_file.getInt();
    Int16 it = vtkToArcaneCellType(vtk_ct, cells_nb_node[i]);
    cells_type[i] = ItemTypeId{ it };
  }
  _readMetadata(mesh, vtk_file);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Lecture des metadata.
 *
 * \param mesh Le maillage à remplir
 * \param vtk_file Référence vers un objet VtkFile
 * \return false si tout s'est bien passé, true sinon
 */
bool VtkMeshIOService::
_readMetadata(IMesh* mesh, VtkFile& vtk_file)
{
  ARCANE_UNUSED(mesh);

  //const char* func_name = "VtkMeshIOService::_readMetadata()";

  if (vtk_file.isEof())
    return false;
  String meta = vtk_file.getNextLine();

  // METADATA ?
  if (!vtk_file.isEqualString(meta, "METADATA")) {
    // S'il n'y a pas de METADATA, on demande à ce que la ligne lue soit relue la prochaine fois.
    vtk_file.reReadSameLine();
    return false;
  }

  // Tant qu'il n'y a pas de ligne vide, on lit.
  while (!vtk_file.isEmptyNextLine() && !vtk_file.isEof()) {
  }
  return false;

  // // Si l'on a besoin de faire quelque chose avec les METADATA un jour, voilà un code non testé.
  // std::string trash;
  // Real trash_real;
  // const char* buf = vtk_file.getNextLine();

  // // INFORMATION ou COMPONENT_NAMES
  // std::istringstream iline(buf);

  // String name_str;
  // String data_type_str;
  // Integer nb_info = 0;
  // Integer size_vector = 0;

  // iline >> ws >> name_str;

  // if(vtk_file.isEqualString(name_str, "INFORMATION"))
  // {
  //   iline >> ws >> nb_info;
  //   for( Integer i = 0; i < nb_info; i++)
  //   {
  //     buf = vtk_file.getNextLine();
  //     std::istringstream iline(buf);

  //     // NAME [key name] LOCATION [key location (e.g. class name)]
  //     iline >> ws >> trash >> ws >> data_type_str >> ws >> trash >> ws >> name_str;

  //     buf = vtk_file.getNextLine();
  //     std::istringstream iline(buf);

  //     if(vtk_file.isEqualString(data_type_str, "StringVector"))
  //     {
  //       iline >> ws >> trash >> ws >> size_vector;
  //       for(Integer j = 0; j < size_vector; j++)
  //       {
  //         trash = vtk_file.getNextLine();
  //       }
  //     }

  //     // else if(vtk_file.isEqualString(data_type_str, "Double")
  //     // || vtk_file.isEqualString(data_type_str, "IdType")
  //     // || vtk_file.isEqualString(data_type_str, "Integer")
  //     // || vtk_file.isEqualString(data_type_str, "UnsignedLong")
  //     // || vtk_file.isEqualString(data_type_str, "String"))
  //     // {
  //     //   iline >> ws >> trash >> ws >> trash_real;
  //     // }

  //     // else if(vtk_file.isEqualString(data_type_str, "DoubleVector")
  //     //      || vtk_file.isEqualString(data_type_str, "IntegerVector")
  //     //      || vtk_file.isEqualString(data_type_str, "StringVector"))
  //     // {
  //     //   iline >> ws >> trash >> ws >> size_vector;
  //     //   for(Integer j = 0; j < size_vector; j++)
  //     //   {
  //     //     iline >> ws >> trash_real;
  //     //   }
  //     // }
  //   }
  // }

  // else if(vtk_file.isEqualString(name_str, "COMPONENT_NAMES"))
  // {
  //   while(!vtk_file.isEmptyNextLine())
  //   {
  //     trash = vtk_file.getCurrentLine();
  //   }
  // }

  // else
  // {
  //   throw IOException(func_name,"Syntax error after METADATA tag");
  // }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Permet de lire un fichier vtk contenant une UNSTRUCTURED_GRID.
 *
 * \param mesh Le maillage à remplir
 * \param vtk_file Référence vers un objet VtkFile
 * \param use_internal_partition Doit-on utiliser le partitionneur interne ou non
 * \return false si tout s'est bien passé, true sinon
 */
bool VtkMeshIOService::
_readUnstructuredGrid(IPrimaryMesh* mesh, VtkFile& vtk_file, bool use_internal_partition)
{
  Integer nb_node = 0;
  Integer nb_cell = 0;
  Integer nb_cell_node = 0;
  Int32 sid = mesh->parallelMng()->commRank();
  UniqueArray<Real3> node_coords;
  UniqueArray<Int32> cells_local_id;

  // Si on utilise le partitionneur interne, seul le sous-domaine lit le maillage
  bool need_read = true;

  if (use_internal_partition)
    need_read = (sid == 0);

  std::array<Int64, 4> nb_cell_by_dimension = {};
  Int32 mesh_dimension = -1;
  ItemTypeMng* itm = mesh->itemTypeMng();
  UnstructuredMeshAllocateBuildInfo mesh_build_info(mesh);

  if (need_read) {
    // Lecture première partie du fichier (après header).
    _readNodesUnstructuredGrid(mesh, vtk_file, node_coords);
    debug() << "Lecture _readNodesUnstructuredGrid OK";
    nb_node = node_coords.size();

    // Lecture des infos des mailles
    // Lecture de la connectivité
    UniqueArray<Integer> cells_nb_node;
    UniqueArray<Int64> cells_connectivity;
    UniqueArray<ItemTypeId> cells_type;
    _readCellsUnstructuredGrid(mesh, vtk_file, cells_nb_node, cells_type, cells_connectivity);
    debug() << "Lecture _readCellsUnstructuredGrid OK";

    nb_cell = cells_nb_node.size();
    nb_cell_node = cells_connectivity.size();
    cells_local_id.resize(nb_cell);

    // Création des mailles
    mesh_build_info.preAllocate(nb_cell, nb_cell_node);

    {
      Int32 connectivity_index = 0;
      for (Integer i = 0; i < nb_cell; ++i) {
        Int32 current_cell_nb_node = cells_nb_node[i];
        Int64 cell_unique_id = i;

        cells_local_id[i] = i;

        Int16 cell_dim = itm->typeFromId(cells_type[i])->dimension();
        if (cell_dim >= 0 && cell_dim <= 3)
          ++nb_cell_by_dimension[cell_dim];

        auto cell_nodes = cells_connectivity.subView(connectivity_index,current_cell_nb_node);
        mesh_build_info.addCell(cells_type[i],cell_unique_id, cell_nodes);
        connectivity_index += current_cell_nb_node;
      }
    }
    // Vérifie qu'on n'a pas de mailles de différentes dimensions
    Int32 nb_different_dim = 0;
    for (Int32 i = 0; i < 4; ++i)
      if (nb_cell_by_dimension[i] != 0) {
        ++nb_different_dim;
        mesh_dimension = i;
      }
    if (nb_different_dim > 1)
      ARCANE_FATAL("The mesh contains cells of different dimension. nb0={0} nb1={1} nb2={2} nb3={3}",
                   nb_cell_by_dimension[0], nb_cell_by_dimension[1], nb_cell_by_dimension[2], nb_cell_by_dimension[3]);
  }

  // Positionne la dimension du maillage.
  {
    Integer wanted_dimension = mesh_dimension;
    wanted_dimension = mesh->parallelMng()->reduce(Parallel::ReduceMax, wanted_dimension);
    //mesh->setDimension(wanted_dimension);
    mesh_build_info.setMeshDimension(wanted_dimension);
  }

  mesh_build_info.allocateMesh();

  // Positionne les coordonnées
  {
    VariableNodeReal3& nodes_coord_var(mesh->nodesCoordinates());
    ENUMERATE_NODE (inode, mesh->allNodes()) {
      Node node = *inode;
      nodes_coord_var[inode] = node_coords[node.uniqueId().asInt32()];
    }
  }

  // Maintenant, regarde s'il existe des données associées aux fichier
  bool r = _readData(mesh, vtk_file, use_internal_partition, IK_Cell, cells_local_id, nb_node);
  debug() << "Lecture _readData OK";

  return r;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Permet de lire le fichier truc.vtkfaces.vtk (s'il existe).
 *
 * \param mesh Le maillage à remplir
 * \param file_name Le nom du fichier vtk (avec l'extension)
 * \param dir_name Le chemin du fichier
 * \param use_internal_partition Doit-on utiliser le partitionneur interne ou non
 */
void VtkMeshIOService::
_readFacesMesh(IMesh* mesh, const String& file_name, const String& dir_name,
               bool use_internal_partition)
{
  ARCANE_UNUSED(dir_name);

  std::ifstream ifile(file_name.localstr(), std::ifstream::binary);
  if (!ifile) {
    info() << "No face descriptor file found '" << file_name << "'";
    return;
  }

  VtkFile vtk_file(&ifile);
  const char* buf = 0;

  // Lecture de la description
  String title = vtk_file.getNextLine();
  info() << "Reading VTK file '" << file_name << "'";
  info() << "Title of VTK file: " << title;

  String format = vtk_file.getNextLine();
  if (VtkFile::isEqualString(format, "BINARY")) {
    vtk_file.setIsBinaryFile(true);
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
    vtk_file.checkString(dataset_str, "DATASET");

    if (VtkFile::isEqualString(mesh_type_str, "UNSTRUCTURED_GRID")) {
      mesh_type = VTK_MT_UnstructuredGrid;
    }

    if (mesh_type == VTK_MT_Unknown) {
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
      need_read = (sid == 0);

    if (need_read) {
      {
        // Lit des noeuds, mais ne conserve pas leur coordonnées car cela n'est
        // pas nécessaire.
        UniqueArray<Real3> node_coords;
        _readNodesUnstructuredGrid(mesh, vtk_file, node_coords);
        //nb_node = node_coords.size();
      }

      // Lecture des infos des faces
      // Lecture de la connectivité
      UniqueArray<Integer> faces_nb_node;
      UniqueArray<Int64> faces_connectivity;
      UniqueArray<ItemTypeId> faces_type;
      _readCellsUnstructuredGrid(mesh, vtk_file, faces_nb_node, faces_type, faces_connectivity);
      nb_face = faces_nb_node.size();
      //nb_face_node = faces_connectivity.size();

      // Il faut à partir de la connectivité retrouver les localId() des faces
      faces_local_id.resize(nb_face);
      {
        IMeshUtilities* mu = mesh->utilities();
        mu->getFacesLocalIdFromConnectivity(faces_type, faces_connectivity, faces_local_id);
      }
    }

    // Maintenant, regarde s'il existe des données associées aux fichiers
    _readData(mesh, vtk_file, use_internal_partition, IK_Face, faces_local_id, 0);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Permet de lire les données complémentaires (POINT_DATA / CELL_DATA).
 *
 * \param mesh Le maillage à remplir
 * \param file_name Le nom du fichier vtk (avec l'extension)
 * \param use_internal_partition Doit-on utiliser le partitionneur interne ou non
 * \param cell_kind Type des cells du maillage
 * \param local_id Tableau contenant les local_id des cells
 * \param nb_node Nombre de nodes
 * \return false si tout s'est bien passé, true sinon
 */
bool VtkMeshIOService::
_readData(IMesh* mesh, VtkFile& vtk_file, bool use_internal_partition,
          eItemKind cell_kind, Int32ConstArrayView local_id, Integer nb_node)
{
  // Seul le sous-domain maitre lit les valeurs. Par contre, les autres
  // sous-domaines doivent connaitre la liste des variables et groupes créées.
  // Si une donnée porte le nom 'GROUP_*', on considère qu'il s'agit d'un
  // groupe

  IParallelMng* pm = mesh->parallelMng();
  IVariableMng* variable_mng = mesh->variableMng();

  Int32 sid = pm->commRank();

  // Si pas de données, retourne immédiatement.
  {
    Byte has_data = 1;
    if ((sid == 0) && vtk_file.isEof())
      has_data = 0;

    ByteArrayView bb(1, &has_data);
    pm->broadcast(bb, 0);
    // Pas de data.
    if (!has_data)
      return false;
  }

  OStringStream created_infos_str;
  created_infos_str() << "<?xml version='1.0' ?>\n";
  created_infos_str() << "<infos>";

  Integer nb_cell_kind = mesh->nbItem(cell_kind);
  const char* buf = 0;

  if (sid == 0) {
    bool reading_node = false;
    bool reading_cell = false;
    while (((buf = vtk_file.getNextLine()) != 0) && !vtk_file.isEof()) {
      debug() << "Read line";
      std::istringstream iline(buf);
      std::string data_str;
      iline >> data_str;

      // Si l'on a un bloc "CELL_DATA".
      if (VtkFile::isEqualString(data_str, "CELL_DATA")) {
        Integer nb_item = 0;
        iline >> ws >> nb_item;
        reading_node = false;
        reading_cell = true;
        if (nb_item != nb_cell_kind)
          error() << "Size expected = " << nb_cell_kind << " found = " << nb_item;
      }

      // Si l'on a un bloc "POINT_DATA".
      else if (VtkFile::isEqualString(data_str, "POINT_DATA")) {
        Integer nb_item = 0;
        iline >> ws >> nb_item;
        reading_node = true;
        reading_cell = false;
        if (nb_item != nb_node)
          error() << "Size expected = " << nb_node << " found = " << nb_item;
      }

      // Si l'on a un bloc "FIELD".
      else if (VtkFile::isEqualString(data_str, "FIELD")) {
        std::string name_str;
        int nb_fields;

        iline >> ws >> name_str >> ws >> nb_fields;

        Integer nb_item = 0;
        std::string type_str;
        std::string s_name_str;
        int nb_component = 1;
        bool is_group = false;

        for (Integer i = 0; i < nb_fields; i++) {
          buf = vtk_file.getNextLine();
          std::istringstream iline(buf);
          iline >> ws >> s_name_str >> ws >> nb_component >> ws >> nb_item >> ws >> type_str;

          if (nb_item != nb_cell_kind && reading_cell && !reading_node)
            error() << "Size expected = " << nb_cell_kind << " found = " << nb_item;

          if (nb_item != nb_node && !reading_cell && reading_node)
            error() << "Size expected = " << nb_node << " found = " << nb_item;

          String name_str = s_name_str;
          String cstr = name_str.substring(0, 6);

          if (cstr == "GROUP_") {
            is_group = true;
            String new_name = name_str.substring(6);
            debug() << "** ** ** GROUP ! name=" << new_name;
            name_str = new_name;
          }

          if (is_group) {
            if (!VtkFile::isEqualString(type_str, "int")) {
              error() << "Group type must be 'int', found=" << type_str;
              return true;
            }

            if (reading_node) {
              created_infos_str() << "<node-group name='" << name_str << "'/>";
              _readNodeGroup(mesh, vtk_file, name_str, nb_node);
            }

            if (reading_cell) {
              created_infos_str() << "<cell-group name='" << name_str << "'/>";
              _readItemGroup(mesh, vtk_file, name_str, nb_cell_kind, cell_kind, local_id);
            }
          }

          // TODO : Voir un exemple si possible.
          else {
            if (!VtkFile::isEqualString(type_str, "float") && !VtkFile::isEqualString(type_str, "double")) {
              error() << "Expecting 'float' or 'double' data type, found=" << type_str;
              return true;
            }

            if (reading_node) {
              fatal() << "Unable to read POINT_DATA: feature not implemented";
            }
            if (reading_cell) {
              created_infos_str() << "<cell-variable name='" << name_str << "'/>";

              if (cell_kind != IK_Cell)
                throw IOException("Unable to read face variables: feature not supported");

              _readCellVariable(mesh, vtk_file, name_str, nb_cell_kind);
            }
          }
        }
      }

      else {
        // Lecture des valeurs (bloc "CELL_DATA" ou "POINT_DATA")
        if (reading_node || reading_cell) {
          std::string type_str;
          std::string s_name_str;
          //String name_str;
          bool is_group = false;
          int nb_component = 1;

          iline >> ws >> s_name_str >> ws >> type_str >> ws >> nb_component;
          debug() << "** ** ** READNAME: name=" << s_name_str << " type=" << type_str;

          String name_str = s_name_str;
          String cstr = name_str.substring(0, 6);

          if (cstr == "GROUP_") {
            is_group = true;
            String new_name = name_str.substring(6);
            info() << "** ** ** GROUP ! name=" << new_name;
            name_str = new_name;
          }

          if (!VtkFile::isEqualString(data_str, "SCALARS")) {
            error() << "Expecting 'SCALARS' data type, found=" << data_str;
            return true;
          }

          if (is_group) {
            if (!VtkFile::isEqualString(type_str, "int")) {
              error() << "Group type must be 'int', found=" << type_str;
              return true;
            }

            // Pour lire LOOKUP_TABLE
            buf = vtk_file.getNextLine();

            if (reading_node) {
              created_infos_str() << "<node-group name='" << name_str << "'/>";
              _readNodeGroup(mesh, vtk_file, name_str, nb_node);
            }

            if (reading_cell) {
              created_infos_str() << "<cell-group name='" << name_str << "'/>";
              _readItemGroup(mesh, vtk_file, name_str, nb_cell_kind, cell_kind, local_id);
            }
          }
          else {
            if (!VtkFile::isEqualString(type_str, "float") && !VtkFile::isEqualString(type_str, "double")) {
              error() << "Expecting 'float' or 'double' data type, found=" << type_str;
              return true;
            }

            // Pour lire LOOKUP_TABLE
            /*buf = */ vtk_file.getNextLine();
            if (reading_node) {
              fatal() << "Unable to read POINT_DATA: feature not implemented";
            }
            if (reading_cell) {
              created_infos_str() << "<cell-variable name='" << name_str << "'/>";

              if (cell_kind != IK_Cell)
                throw IOException("Unable to read face variables: feature not supported");

              _readCellVariable(mesh, vtk_file, name_str, nb_cell_kind);
            }
          }
        }
        else {
          error() << "Expecting value CELL_DATA or POINT_DATA, found='" << data_str << "'";
          return true;
        }
      }
    }
  }
  created_infos_str() << "</infos>";
  if (use_internal_partition) {
    ByteUniqueArray bytes;
    if (sid == 0) {
      String str = created_infos_str.str();
      ByteConstArrayView bv = str.utf8();
      Integer len = bv.size();
      bytes.resize(len + 1);
      bytes.copy(bv);
    }

    pm->broadcastMemoryBuffer(bytes, 0);

    if (sid != 0) {
      String str = String::fromUtf8(bytes);
      info() << "FOUND STR=" << bytes.size() << " " << str;
      ScopedPtrT<IXmlDocumentHolder> doc(IXmlDocumentHolder::loadFromBuffer(bytes, "InternalBuffer", traceMng()));
      XmlNode doc_node = doc->documentNode();

      // Lecture des variables
      {
        XmlNodeList vars = doc_node.documentElement().children("cell-variable");
        for (XmlNode xnode : vars) {
          String name = xnode.attrValue("name");
          info() << "Building variable: " << name;
          VariableCellReal* var = new VariableCellReal(VariableBuildInfo(mesh, name));
          variable_mng->_internalApi()->addAutoDestroyVariable(var);
        }
      }

      // Lecture des groupes de mailles
      {
        XmlNodeList vars = doc_node.documentElement().children("cell-group");
        IItemFamily* cell_family = mesh->itemFamily(cell_kind);
        for (XmlNode xnode : vars) {
          String name = xnode.attrValue("name");
          info() << "Building group: " << name;
          cell_family->createGroup(name);
        }
      }

      // Lecture des groupes de noeuds
      {
        XmlNodeList vars = doc_node.documentElement().children("node-group");
        IItemFamily* node_family = mesh->nodeFamily();
        for (XmlNode xnode : vars) {
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

/*!
 * \brief Permet de créer un groupe de face de nom "name" et composé des faces ayant les ids inclus dans "faces_lid".
 *
 * \param mesh Le maillage à remplir
 * \param name Le nom du groupe à créer
 * \param faces_lid Les ids des faces à inclure dans le groupe
 */
void VtkMeshIOService::
_createFaceGroup(IMesh* mesh, const String& name, Int32ConstArrayView faces_lid)
{
  info() << "Building face group '" << name << "'"
         << " size=" << faces_lid.size();

  mesh->faceFamily()->createGroup(name, faces_lid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Permet de créer une variable aux mailles à partir des infos du fichier vtk.
 *
 * \param mesh Le maillage à remplir
 * \param vtk_file Référence vers un objet VtkFile
 * \param var_name Le nom de la variable à créer
 * \param nb_cell Le nombre de cells
 */
void VtkMeshIOService::
_readCellVariable(IMesh* mesh, VtkFile& vtk_file, const String& var_name, Integer nb_cell)
{
  //TODO Faire la conversion uniqueId() vers localId() correcte
  info() << "Reading values for variable: " << var_name << " n=" << nb_cell;
  auto* var = new VariableCellReal(VariableBuildInfo(mesh, var_name));
  mesh->variableMng()->_internalApi()->addAutoDestroyVariable(var);
  RealArrayView values(var->asArray());
  for (Integer i = 0; i < nb_cell; ++i) {
    Real v = vtk_file.getDouble();
    values[i] = v;
  }
  _readMetadata(mesh, vtk_file);
  info() << "Variable build finished: " << vtk_file.isEof();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Permet de créer un groupe d'item.
 *
 * \param mesh Le maillage à remplir
 * \param vtk_file Référence vers un objet VtkFile
 * \param name Le nom du groupe à créer
 * \param nb_item Nombre d'items à lire et à inclure dans le groupe
 * \param ik Type des items lus
 * \param local_id Tableau contenant les local_id des cells
 */
void VtkMeshIOService::
_readItemGroup(IMesh* mesh, VtkFile& vtk_file, const String& name, Integer nb_item,
               eItemKind ik, Int32ConstArrayView local_id)
{
  IItemFamily* item_family = mesh->itemFamily(ik);
  info() << "Reading group info for group: " << name;

  Int32UniqueArray ids;
  for (Integer i = 0; i < nb_item; ++i) {
    Integer v = vtk_file.getInt();
    if (v != 0)
      ids.add(local_id[i]);
  }
  info() << "Building group: " << name << " nb_element=" << ids.size();

  item_family->createGroup(name, ids);

  _readMetadata(mesh, vtk_file);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Permet de créer un groupe de node.
 *
 * \param mesh Le maillage à remplir
 * \param vtk_file Référence vers un objet VtkFile
 * \param name Le nom du groupe à créer
 * \param nb_item Nombre d'items à lire et à inclure dans le groupe
 */
void VtkMeshIOService::
_readNodeGroup(IMesh* mesh, VtkFile& vtk_file, const String& name, Integer nb_item)
{
  IItemFamily* item_family = mesh->itemFamily(IK_Node);
  info() << "Lecture infos groupes de noeuds pour le groupe: " << name;

  Int32UniqueArray ids;
  for (Integer i = 0; i < nb_item; ++i) {
    Integer v = vtk_file.getInt();
    if (v != 0)
      ids.add(i);
  }
  info() << "Création groupe: " << name << " nb_element=" << ids.size();

  item_family->createGroup(name, ids);

  _readMetadata(mesh, vtk_file);
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

  explicit VtkLegacyMeshWriter(const ServiceBuildInfo& sbi)
  : BasicService(sbi)
  {}

 public:

  void build() override {}

 public:

  bool writeMeshToFile(IMesh* mesh, const String& file_name) override;

 private:

  void _writeMeshToFile(IMesh* mesh, const String& file_name, eItemKind cell_kind);
  void _saveGroups(IItemFamily* family, std::ostream& ofile);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(VtkLegacyMeshWriter,
                        ServiceProperty("VtkLegacyMeshWriter", ST_SubDomain),
                        ARCANE_SERVICE_INTERFACE(IMeshWriter));

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
writeMeshToFile(IMesh* mesh, const String& file_name)
{
  String fname = file_name;
  // Ajoute l'extension '.vtk' si elle n'y est pas.
  if (!fname.endsWith(".vtk"))
    fname = fname + ".vtk";
  _writeMeshToFile(mesh, fname, IK_Cell);
  _writeMeshToFile(mesh, fname + "faces.vtk", IK_Face);
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
_writeMeshToFile(IMesh* mesh, const String& file_name, eItemKind cell_kind)
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

  // Sauve les nœuds
  {
    ofile << "POINTS " << nb_node << " double\n";
    VariableNodeReal3& coords(mesh->toPrimaryMesh()->nodesCoordinates());
    Integer node_index = 0;
    ENUMERATE_NODE (inode, mesh->allNodes()) {
      Node node = *inode;
      nodes_local_id_to_current[node.localId()] = node_index;
      Real3 xyz = coords[inode];
      ofile << xyz.x << ' ' << xyz.y << ' ' << xyz.z << '\n';
      ++node_index;
    }
  }

  // Sauve les mailles ou faces
  {
    Integer nb_node_cell_kind = nb_cell_kind;
    ENUMERATE_ITEMWITHNODES(iitem, cell_kind_family->allItems())
    {
      nb_node_cell_kind += (*iitem).nbNode();
    }
    ofile << "CELLS " << nb_cell_kind << ' ' << nb_node_cell_kind << "\n";
    ENUMERATE_ITEMWITHNODES(iitem, cell_kind_family->allItems())
    {
      ItemWithNodes item = *iitem;
      Integer item_nb_node = item.nbNode();
      ofile << item_nb_node;
      for (NodeLocalId node_id : item.nodes() ) {
        ofile << ' ' << nodes_local_id_to_current[node_id];
      }
      ofile << '\n';
    }
    // Le type doit être coherent avec celui de vtkCellType.h
    ofile << "CELL_TYPES " << nb_cell_kind << "\n";
    ENUMERATE_ (ItemWithNodes, iitem, cell_kind_family->allItems()) {
      int type = arcaneToVtkCellType(iitem->typeInfo());
      ofile << type << '\n';
    }
  }

  // Si on est dans le maillage des mailles, sauve les groupes de noeuds.
  if (cell_kind == IK_Cell) {
    ofile << "POINT_DATA " << nb_node << "\n";
    _saveGroups(mesh->itemFamily(IK_Node), ofile);
  }

  // Sauve les groupes de mailles
  ofile << "CELL_DATA " << nb_cell_kind << "\n";
  _saveGroups(mesh->itemFamily(cell_kind), ofile);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkLegacyMeshWriter::
_saveGroups(IItemFamily* family, std::ostream& ofile)
{
  info() << "Saving groups for family name=" << family->name();
  UniqueArray<char> in_group_list(family->maxLocalId());
  for (ItemGroupCollection::Enumerator igroup(family->groups()); ++igroup;) {
    ItemGroup group = *igroup;
    // Inutile de sauver le groupe de toutes les entités
    if (group == family->allItems())
      continue;
    //HACK: a supprimer
    if (group.name() == "OuterFaces")
      continue;
    ofile << "SCALARS GROUP_" << group.name() << " int 1\n";
    ofile << "LOOKUP_TABLE default\n";
    in_group_list.fill('0');
    ENUMERATE_ITEM (iitem, group) {
      in_group_list[(*iitem).localId()] = '1';
    }
    ENUMERATE_ITEM (iitem, family->allItems()) {
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
  : AbstractService(sbi)
  {}

 public:

  bool allowExtension(const String& str) override { return str == "vtk"; }
  eReturnType readMeshFromFile(IPrimaryMesh* mesh, const XmlNode& mesh_node, const String& file_name,
                               const String& dir_name, bool use_internal_partition) override

  {
    ARCANE_UNUSED(mesh_node);
    VtkMeshIOService vtk_service(traceMng());
    bool ret = vtk_service.readMesh(mesh, file_name, dir_name, use_internal_partition);
    if (ret)
      return RTError;

    return RTOk;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SUB_DOMAIN_FACTORY(VtkMeshReader, IMeshReader, VtkMeshIO);

ARCANE_REGISTER_SERVICE(VtkMeshReader,
                        ServiceProperty("VtkLegacyMeshReader", ST_SubDomain),
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
      VtkMeshIOService vtk_service(m_trace_mng);
      String fname = m_read_info.fileName();
      m_trace_mng->info() << "VtkLegacy Reader (ICaseMeshReader) file_name=" << fname;
      bool ret = vtk_service.readMesh(pm, fname, m_read_info.directoryName(), m_read_info.isParallelRead());
      if (ret)
        ARCANE_FATAL("Can not read VTK File");
    }

   private:

    ITraceMng* m_trace_mng;
    CaseMeshReaderReadInfo m_read_info;
  };

 public:

  explicit VtkLegacyCaseMeshReader(const ServiceBuildInfo& sbi)
  : AbstractService(sbi)
  {}

 public:

  Ref<IMeshBuilder> createBuilder(const CaseMeshReaderReadInfo& read_info) const override
  {
    IMeshBuilder* builder = nullptr;
    if (read_info.format() == "vtk")
      builder = new Builder(traceMng(), read_info);
    return makeRef(builder);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(VtkLegacyCaseMeshReader,
                        ServiceProperty("VtkLegacyCaseMeshReader", ST_SubDomain),
                        ARCANE_SERVICE_INTERFACE(ICaseMeshReader));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
