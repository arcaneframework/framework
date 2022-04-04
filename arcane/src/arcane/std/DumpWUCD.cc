// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DumpWUCD.cc                                                 (C) 2000-2021 */
/*                                                                           */
/* Exportations des fichiers au format UCD.                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/utils/Enumerator.h"
#include "arcane/utils/Iterator.h"
#include "arcane/utils/List.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/Iostream.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/Item.h"
#include "arcane/ItemEnumerator.h"
#include "arcane/IVariable.h"
#include "arcane/ISubDomain.h"
#include "arcane/IMesh.h"
#include "arcane/IMeshSubMeshTransition.h"
#include "arcane/StdNum.h"
#include "arcane/ItemGroup.h"
#include "arcane/IParallelMng.h"
#include "arcane/Directory.h"
#include "arcane/PostProcessorWriterBase.h"
#include "arcane/Service.h"
#include "arcane/SimpleProperty.h"
#include "arcane/FactoryService.h"
#include "arcane/VariableCollection.h"

#include "arcane/std/DumpW.h"

#include <string>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
// NOTE: Ce format ne fonctionne que en séquentiel

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
Integer code_hex[8]   = { 0, 3, 2, 1, 4, 7, 6, 5 };
Integer code_prism[6] = { 2, 1, 0, 5, 4, 3 };
Integer code_pyr[5]   = { 4, 1, 2, 3, 0 };
Integer code_tet[4]   = { 0, 1, 3, 2 };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 \brief Ecriture au format UCD.
*/
class DumpWUCD
: public TraceAccessor
, public DumpW
{
 public:

  DumpWUCD(ISubDomain* sd,IMesh* mesh, const String& filename,
           RealConstArrayView times, VariableCollection variables);
  ~DumpWUCD();

  void setMetaData(const String& meta_data) override
  {
    ARCANE_UNUSED(meta_data);
  }
  String metaData() const { return String(); }

  void writeVal(IVariable&,ConstArrayView<Byte>) override {}
  void writeVal(IVariable&,ConstArrayView<Real>) override;
  void writeVal(IVariable&,ConstArrayView<Real2>) override {}
  void writeVal(IVariable&,ConstArrayView<Real3>) override;
  void writeVal(IVariable&,ConstArrayView<Int64>) override {}
  void writeVal(IVariable&,ConstArrayView<Int32>) override {}
  void writeVal(IVariable&,ConstArrayView<Real2x2>) override {}
  void writeVal(IVariable&,ConstArrayView<Real3x3>) override {}
  void writeVal(IVariable&,ConstArrayView<String>) override {}

  void writeVal(IVariable&,ConstArray2View<Byte>) override {}
  void writeVal(IVariable&,ConstArray2View<Real>) override {}
  void writeVal(IVariable&,ConstArray2View<Int64>) override {}
  void writeVal(IVariable&,ConstArray2View<Int32>) override {}
  void writeVal(IVariable&,ConstArray2View<Real2>) override {}
  void writeVal(IVariable&,ConstArray2View<Real3>) override {}
  void writeVal(IVariable&,ConstArray2View<Real2x2>) override {}
  void writeVal(IVariable&,ConstArray2View<Real3x3>) override {}

  void writeVal(IVariable&,ConstMultiArray2View<Byte>) override {}
  void writeVal(IVariable&,ConstMultiArray2View<Real>) override {}
  void writeVal(IVariable&,ConstMultiArray2View<Int64>) override {}
  void writeVal(IVariable&,ConstMultiArray2View<Int32>) override {}
  void writeVal(IVariable&,ConstMultiArray2View<Real2>) override {}
  void writeVal(IVariable&,ConstMultiArray2View<Real3>) override {}
  void writeVal(IVariable&,ConstMultiArray2View<Real2x2>) override {}
  void writeVal(IVariable&,ConstMultiArray2View<Real3x3>) override {}

  void beginWrite() override;
  void endWrite() override;
	
 private:

  static constexpr Integer m_max_digit = 5;
  // Nombre de chiffres significatifs pour les afficher les réels.
  static constexpr Integer MAX_FLOAT_DIGIT = FloatInfo<Real>::maxDigit()+1;

  ISubDomain* m_sub_domain;
  IMesh* m_mesh; //!< Maillage
  Directory m_base_directory; //!< Nom du répertoire de stockage
  RealUniqueArray m_times; //!< Liste des instants de temps
  VariableList m_save_variables; //!< Liste des variables a exporter

  UniqueArray<Ref<OStringStream>> m_cell_streams; //!< Valeur des var. aux mailles
  UniqueArray<Ref<OStringStream>> m_node_streams; //!< Valeur des var. aux noeuds
  UniqueArray<Cell> m_managed_cells; //!< Liste des mailles gerees
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DumpWUCD::
DumpWUCD(ISubDomain* sd,IMesh* mesh,const String& filename,RealConstArrayView times,
         VariableCollection variables)
: TraceAccessor(mesh->traceMng())
, m_sub_domain(sd)
, m_mesh(mesh)
, m_base_directory(filename)
, m_times(times)
, m_save_variables(variables.clone())
{
  m_base_directory.createDirectory();

  // filtrage des cellules
  // ne sont gardees que les mailles dont le type est reconnue dans UCD
  ENUMERATE_CELL(it,m_mesh->allCells()){
    Cell cell = *it;
    const int type = cell.type();
    if (type==IT_Vertex || type==IT_Line2 || type==IT_Triangle3
        || type== IT_Quad4 || type==IT_Hexaedron8 || type==IT_Pyramid5
        || type==IT_Pentagon5 || type==IT_Tetraedron4)
      m_managed_cells.add(cell);
    else
      info() << "** Warning: cell type " << cell.uniqueId()
             << " is unknown in UCD format. Cell will be ignored.";
  }

  Integer nb_cell_var = m_managed_cells.size();
  Integer nb_node_var = m_mesh->nbNode();
  m_cell_streams.resize(nb_cell_var);
  m_node_streams.resize(nb_node_var);
  for( Integer i=0; i<nb_cell_var; ++i ){
    m_cell_streams[i] = makeRef(new OStringStream());
    m_cell_streams[i]->stream().precision(MAX_FLOAT_DIGIT);
    m_cell_streams[i]->stream().flags(std::ios::scientific);
  }
  for( Integer i=0; i<nb_node_var; ++i ){
    m_node_streams[i] = makeRef(new OStringStream());
    m_node_streams[i]->stream().precision(MAX_FLOAT_DIGIT);
    m_node_streams[i]->stream().flags(std::ios::scientific);
  }

  debug() << "DumpWUCD::DumpWUCD - " 
          << m_mesh->nbCell() << " cells among which " 
          << m_managed_cells.size() << " have a known type";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DumpWUCD::
~DumpWUCD()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Sauvegarde des variables scalaires.
 * La variable est sauvegardee dans un flux different suivant son origine
 * (noeud ou maille). 
 */
void DumpWUCD::
writeVal(IVariable& v,ConstArrayView<Real> ptr)
{
  info() << "** HERE dump Real variable " << v.name();

  Integer size;
  switch(v.itemKind()) 
  {
  case (IK_Node):
    size = ptr.size();
    for( Integer i=0 ; i<size ; i++)
      m_node_streams[i]->stream() << " " << ptr[i];
    break;
  case (IK_Cell):
    size = m_managed_cells.size();
    for( Integer i=0 ; i<size ; i++){
      const Cell& cell = m_managed_cells[i];
      m_cell_streams[i]->stream() << " " << ptr[cell.localId()];
    }
    break;
  default:
    info() << "Variable not managed by UCD writer: " << v.name();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Sauvegarde des variables vectorielles.
 * La variable est sauvegardee dans un flux different suivant son origine
 * (noeud ou maille). 
 */
void DumpWUCD::
writeVal(IVariable& v,ConstArrayView<Real3> ptr)
{
  info() << "** HERE dump Real3 variable " << v.name();
  
  Integer size;
  switch(v.itemKind()) 
  {
  case (IK_Node):
    size = ptr.size();
    for( Integer i=0 ; i<size ; i++) {
      m_node_streams[i]->stream() << " " << ptr[i].x
                                  << " " << ptr[i].y
                                  << " " << ptr[i].z;
    }
    break;
  case (IK_Cell):
    size = m_managed_cells.size();
    for( Integer i=0 ; i<size ; i++){
      const Cell cell = m_managed_cells[i];
      Integer id = cell.localId();
      m_cell_streams[i]->stream() << " " << ptr[id].x
                                  << " " << ptr[id].y
                                  << " " << ptr[id].z;
    }
    break;
  default:
    info() << "Variable not managed by UCD writer: " << v.name();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Creation du fichier UCD (nomme UCD_<no_iteration>) et de son entete.
 * Cette entete contient :
 * <ul><li>nombre de noeuds, nombre de mailles...</li>
 * <li>coordonnees des noeuds</li>
 * <lidefinition des cellules</li></ul>
 * Notons que le format UCD impose que les donnees des noeuds precedent celles
 * des mailles. Les donnees des mailles sont donc ecrites dans un buffer 
 * temporaire et concatenees au fichier a la fin (methode writeEnd).
 */
void DumpWUCD::
beginWrite()
{
  info() << "** Entering method DumpWUCD::writeBegin";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Concatenation du flux contenant les donnees des mailles
 * au fichier principal.
 */
void DumpWUCD::
endWrite()
{
  info() << "** Entering method DumpWUCD::writeEnd";

  // Creation du nom de fichier UCD : UCD_<no_iteration>
  OStringStream ostr;
  ostr() << "UCD_";
  ostr().fill('0');
  ostr().width(m_max_digit);
  ostr() << m_times.size()-1;
  ostr() << ".inp\0";
  String buf = m_base_directory.file(ostr.str());
  std::ofstream ucd_file(buf.localstr());

  // Ajout de l'entete du fichier
  // Comptage du nombre de donnees aux noeuds et aux mailles et de leurs 
  // tailles. Les seules methodes "write" implementees sont :
  //    - void writeVal(IVariable& v,ConstArrayView<Real3> a)
  //    - void writeVal(IVariable& v,ConstArrayView<Real> a)
  // Par consequent, les seules variables prises en compte sont de type
  // "Real" et "Real3" et de dimension 1.
  IMesh* mesh = m_mesh;
  Integer nb_comp_node_data = 0;
  Integer nb_comp_cell_data = 0;
  Integer comp_node_data_size = 0;
  Integer comp_cell_data_size = 0;
  OStringStream ndata_size_stream, cdata_size_stream;
  OStringStream ndata_name_stream, cdata_name_stream;
  ndata_size_stream().precision(MAX_FLOAT_DIGIT);
  ndata_name_stream().precision(MAX_FLOAT_DIGIT);
  cdata_size_stream().precision(MAX_FLOAT_DIGIT);
  cdata_name_stream().precision(MAX_FLOAT_DIGIT);

  ndata_size_stream().flags(std::ios::scientific);
  ndata_name_stream().flags(std::ios::scientific);
  cdata_size_stream().flags(std::ios::scientific);
  cdata_name_stream().flags(std::ios::scientific);

  for(VariableList::Enumerator i(m_save_variables); ++i; ){
    IVariable* var = *i;

    if (var->dimension() == 1){
      eDataType type = var->dataType();
      eItemKind kind = var->itemKind();
      String name = var->name();
      if (type == DT_Real){
        if (kind == IK_Node){
          debug() << "  Variable " << name 
                << " kind = IK_Node, type = DT_Real";
          nb_comp_node_data++;
          comp_node_data_size++;
          ndata_size_stream() << " 1";  // 1 = taille du Real
          ndata_name_stream() << name << ", Unknown" << '\n';
        }
        else if (kind == IK_Cell){
          debug() << "  Variable " << name 
                << " kind = IK_Cell, type = DT_Real";
          nb_comp_cell_data++;
          comp_cell_data_size++;
          cdata_size_stream() << " 1"; // 1 = taille du Real1
          cdata_name_stream() << name << ", Unknown" << '\n';
        }
      }
      else if (type == DT_Real3){
        if (kind == IK_Node){
          debug() << "  Variable " << name
                << " kind = IK_Node, type = DT_Real3";
          nb_comp_node_data++;
          comp_node_data_size+=3;
          ndata_size_stream() << " 3";  // 3 = taille du Real3
          ndata_name_stream() << name << ", Unknown" << '\n';
        }
        else if (kind == IK_Cell){
          debug() << "  Variable " << name
                << " kind = IK_Cell, type = DT_Real3";
          nb_comp_cell_data++;
          comp_cell_data_size+=3;
          cdata_size_stream() << " 3";  // 3 = taille du Real3
          cdata_name_stream() << name << ", Unknown" << '\n';
        }
      }
    }
  }

  Integer nb_node = mesh->nbNode();
  Integer nb_managed_cell = m_managed_cells.size();
  ucd_file << nb_node << " " 
           << nb_managed_cell << " " 
           << comp_node_data_size << " " 
           << comp_cell_data_size << " 0" << '\n';

  // ajout des coordonnees des noeuds
  ConstArrayView<Real3> node_coords = mesh->toPrimaryMesh()->nodesCoordinates().asArray();
  for( Integer i=0 ; i<nb_node ; i++){
    const Real3 node_coord = node_coords[i];
    ucd_file << i+1 << " " 
             << node_coord.x << " " 
             << node_coord.y << " " 
             << node_coord.z << '\n';
  }

  // ajout de la description des mailles
  for( Integer iz=0 ; iz<nb_managed_cell ; ++iz ){
    const Cell cell = m_managed_cells[iz];
    Integer id = cell.localId();
    ucd_file << id+1 << " 1 ";
    Integer nb_cell_node=cell.nbNode();
    switch(cell.type()){
    case(IT_Vertex):
      ucd_file << "pt";
      ucd_file << " " << cell.node(0).localId()+1;
      break;
    case(IT_Line2):
      ucd_file << "line";
      for( Integer i=0 ; i<nb_cell_node ; i++) 
        ucd_file << " " << cell.node(i).localId()+1;
      break;
    case(IT_Triangle3):
      ucd_file << "tri";
      for( Integer i=0 ; i<nb_cell_node ; i++) 
        ucd_file << " " << cell.node(i).localId()+1;
      break;
    case(IT_Quad4):
      ucd_file << "quad";
      for( Integer i=0 ; i<nb_cell_node ; i++) 
        ucd_file << " " << cell.node(i).localId()+1;
      break;
    case(IT_Hexaedron8):
      ucd_file << "hex";
      for( Integer i=0 ; i<nb_cell_node ; i++) 
        ucd_file << " " << cell.node(code_hex[i]).localId()+1;
      break;
    case(IT_Pyramid5):
      ucd_file << "pyr";
      for( Integer i=0 ; i<nb_cell_node ; i++) 
        ucd_file << " " << cell.node(code_pyr[i]).localId()+1;
      break;
    case(IT_Pentagon5):
      ucd_file << "prism";
      for( Integer i=0 ; i<nb_cell_node ; i++) 
        ucd_file << " " << cell.node(code_prism[i]).localId()+1;
      break;
    case(IT_Tetraedron4):
      ucd_file << "tet";
      for( Integer i=0 ; i<nb_cell_node ; i++) 
        ucd_file << " " << cell.node(code_tet[i]).localId()+1;
      break;
    default:
      // ce cas ne peut pas arriver car filtrage dans le constructeur
      break;
    }

    ucd_file << '\n';
  }

  // ajout du nombre de donnees aux noeuds, de leur taille, de leur nom
  // et des valeurs des variables
  if (nb_comp_node_data){
    ucd_file << nb_comp_node_data 
             << ndata_size_stream.str() 
             << '\n'
             << ndata_name_stream.str();
    for( Integer i=0 ; i<nb_node ; i++)
      ucd_file << i+1 << m_node_streams[i]->str() << '\n';
  }

  // idem pour les mailles
  if (nb_comp_cell_data){
    ucd_file << nb_comp_cell_data 
             << cdata_size_stream.str() 
             << '\n'
             << cdata_name_stream.str();
    for( Integer i=0 ; i<nb_managed_cell ; i++){
      const Cell& cell = m_managed_cells[i];
      ucd_file << cell.localId()+1 << m_cell_streams[i]->str() << '\n';
    }
  }

  OStringStream code_ostr;
  code_ostr() << "U_";
  code_ostr().fill('0');
  code_ostr().width(m_max_digit);
  code_ostr() << m_times.size()-1;
  code_ostr() << '\0';
  buf = m_base_directory.file(code_ostr.str());
  std::ofstream code_file(buf.localstr());
  code_file << m_sub_domain->commonVariables().globalIteration() << '\n'
                 << m_times[m_times.size()-1] << '\n'
                 << "3" << '\n' << "1" << '\n'
                 << "7" << '\n' << "Unknown" << '\n'
                 << comp_node_data_size << '\n' << comp_cell_data_size << '\n';
  for( Integer i=0 ; i<comp_node_data_size  ; i++)
    code_file << "1"<< '\n';
  for( Integer i=0 ; i<comp_cell_data_size  ; i++)
    code_file << "1"<< '\n';
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Post-traitement au format UCD.
 */
class UCDPostProcessorService
: public PostProcessorWriterBase
{
 public:
  explicit UCDPostProcessorService(const ServiceBuildInfo& sbi)
  : PostProcessorWriterBase(sbi), m_writer(nullptr) {}
  IDataWriter* dataWriter() override { return m_writer; }
  void notifyBeginWrite() override;
  void notifyEndWrite() override;
  void close() override {}
 private:
  DumpW* m_writer;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UCDPostProcessorService::
notifyBeginWrite()
{
  m_writer = new DumpWUCD(subDomain(),subDomain()->defaultMesh(),
                          baseDirectoryName(),times(),variables());
}

void UCDPostProcessorService::
notifyEndWrite()
{
  delete m_writer;
  m_writer = 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_CASE_OPTIONS_NOAXL_FACTORY(UCDPostProcessorService,IPostProcessorWriter,UCDPostProcessor);

ARCANE_REGISTER_SUB_DOMAIN_FACTORY(UCDPostProcessorService,IPostProcessorWriter,UCDPostProcessor);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
