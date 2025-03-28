// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneCasePartitioner.cc                                    (C) 2000-2024 */
/*                                                                           */
/* Service de partitionnement externe du maillage.                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ITraceMng.h"

#include "arcane/core/BasicService.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/ServiceFinder2.h"
#include "arcane/core/FactoryService.h"
#include "arcane/core/SerializeBuffer.h"
#include "arcane/core/IMeshPartitioner.h"
#include "arcane/core/IMainFactory.h"
#include "arcane/core/IMeshModifier.h"
#include "arcane/core/Properties.h"
#include "arcane/core/IInitialPartitioner.h"
#include "arcane/core/Timer.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IMeshSubMeshTransition.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/IDirectExecution.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/IMeshUtilities.h"
#include "arcane/core/IMeshWriter.h"
#include "arcane/core/ITimeStats.h"
#include "arcane/core/ServiceBuilder.h"
#include "arcane/core/IMeshPartitionConstraintMng.h"
#include "arcane/core/ExternalPartitionConstraint.h"

#include "arcane/std/ArcaneCasePartitioner_axl.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ArcaneCasePartitioner;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ArcaneInitialPartitioner
: public IInitialPartitioner
{
 public:

  struct TrueOwnerInfo
  {
    VariableCellInt32* m_true_cells_owner = nullptr;
    VariableNodeInt32* m_true_nodes_owner = nullptr;
  };

 public:

  ArcaneInitialPartitioner(ArcaneCasePartitioner* mt,ISubDomain* sd)
  : m_sub_domain(sd)
  , m_main(mt)
  {
  }
  void build() override {}
  void partitionAndDistributeMeshes(ConstArrayView<IMesh*> meshes) override;

 private:

  //! Regroupe les mailles associées aux contraintes sur un même proc
  void _mergeConstraints(ConstArrayView<IMesh*> meshes);

  //! Affiche des statistiques sur le partitionnement
  void _printStats(Integer nb_part,IMesh* mesh,VariableCellInt32& new_owners);

 public:

  ISubDomain* m_sub_domain = nullptr;
  ArcaneCasePartitioner* m_main = nullptr;
  //! Stocke pour chaque maillage une variable indiquant pour chaque maille quelle partie la possède.
  UniqueArray<TrueOwnerInfo> m_part_indexes;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service de partitionnement externe du maillage.
 */
class ArcaneCasePartitioner
: public ArcaneArcaneCasePartitionerObject
{
 public:

 public:

  explicit ArcaneCasePartitioner(const ServiceBuildInfo& sbi);
  ~ArcaneCasePartitioner() override;

 public:

  void build() override {}
  void execute() override;
  void setParallelMng(IParallelMng*) override {}
  bool isActive() const override { return true; }

 private:

  //! Ouverture du fichier Correspondance (seulement sur le proc 0)
  void _initCorrespondance(Int32 my_rank);
  
  //! Ecriture du fichier Correspondance
  void _writeCorrespondance(Int32 rank, Int64Array& nodesUniqueId, Int64Array& cellsUniqueId);

  //! Fermeture du fichier Correspondance (seulement sur le proc 0)
  void _finalizeCorrespondance(Int32 my_rank);


 private:

  std::ofstream m_sortiesCorrespondance;

  ArcaneInitialPartitioner* m_init_part = nullptr;

  void _partitionMesh(Int32 nb_part);
  void _computeGroups(IItemFamily* current_family,IItemFamily* new_family);
  void _addGhostLayers(CellGroup current_all_cells, Array<Cell>& cells_selected_for_new_mesh,
                       Integer nb_layer,Integer maxLocalIdCell, Integer maxLocalIdNode);
  void _addGhostGroups(IMesh* new_mesh, Array<Cell>& cells_selected_for_new_mesh,
                       VariableCellInt32& true_cells_owner, VariableNodeInt32& true_nodes_owner, 
                       Int32Array& new_cells_local_id, Integer id_loc);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneInitialPartitioner::
_mergeConstraints(ConstArrayView<IMesh*> meshes)
{
  Integer nb_mesh = meshes.size();
  if (nb_mesh!=1)
    ARCANE_FATAL("Can not partition multiple meshes");

  IMesh* mesh = meshes[0];
  ISubDomain* sd = m_sub_domain;
  ITraceMng* tm = sd->traceMng();

  tm->info()<<" _regroupeContraintes: nbMailles = "<<meshes[0]->nbCell() <<", nbMaillesLocales = "<< meshes[0]->ownCells().size();

  Integer nb_contraintes = m_main->options()->constraints.size();
  tm->info() << "Number of constraints = " << nb_contraintes;
  if (nb_contraintes==0)
    return;
 
  IItemFamily* current_cell_family = mesh->cellFamily();
  VariableItemInt32& cells_new_owner(current_cell_family->itemsNewOwner());
  ENUMERATE_CELL(icell,current_cell_family->allItems()){
    cells_new_owner[icell] = (*icell).owner();
  }

  sd->timeStats()->dumpTimeAndMemoryUsage(sd->parallelMng());
  IMeshPartitionConstraint* c = new ExternalPartitionConstraint(mesh, m_main->options()->constraints);
  mesh->partitionConstraintMng()->addConstraint(c);
  mesh->partitionConstraintMng()->computeAndApplyConstraints();

  cells_new_owner.synchronize();
  mesh->utilities()->changeOwnersFromCells();  
  mesh->modifier()->setDynamic(true);
  bool compact = mesh->properties()->getBool("compact");
  mesh->properties()->setBool("compact", true);
  mesh->toPrimaryMesh()->exchangeItems();
  mesh->properties()->setBool("compact", compact);
#if 0
#ifdef ARCANE_DEBUG
  ScopedPtrT<IMeshWriter> mesh_writer;
  FactoryT<IMeshWriter> mesh_writer_factory(sd->serviceMng());
  mesh_writer = mesh_writer_factory.createInstance("Lima",true);
  IParallelMng* pm = sd->parallelMng();
  Int32 my_rank = pm->commRank();

  StringBuilder filename = "cut_mesh_after_";
  filename += my_rank;
  filename += ".mli2";
  mesh_writer->writeMeshToFile(mesh,filename);
#endif
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneInitialPartitioner::
partitionAndDistributeMeshes(ConstArrayView<IMesh*> meshes)
{
  String lib_name = m_main->options()->library();
  ISubDomain* sd = m_sub_domain;
  IParallelMng* pm = sd->parallelMng();
  Int32 nb_rank = pm->commSize();
  //Int32 my_rank = pm->commRank();
  ServiceBuilder<IMeshPartitioner> service_builder(sd);
  auto mesh_partitioner(service_builder.createReference(lib_name,SB_AllowNull));
  ITraceMng* tm = sd->traceMng();
  tm->info() << "DoInitialPartition. Service=" << lib_name;

  if (!mesh_partitioner.get())
    ARCANE_FATAL("can not found service named '{0}' for initial mesh partitioning", lib_name);

  _mergeConstraints(meshes);

  Integer nb_mesh = meshes.size();
  if (nb_mesh!=1)
    ARCANE_FATAL("Can not partition multiple meshes");

  m_part_indexes.resize(nb_mesh);
  Int32 nb_part = m_main->options()->nbCutPart();
  if (nb_part==0)
    nb_part = nb_rank;
  tm->info() << "NbPart = " << nb_part << " nb_mesh=" << nb_mesh;

  for( Integer i=0; i<nb_mesh; ++i ){
    IMesh* mesh = meshes[i];
    ARCANE_CHECK_POINTER(mesh);
    VariableCellInt32* p_true_cells_owner = new VariableCellInt32(VariableBuildInfo(mesh,"TrueCellsOwner"));
    VariableNodeInt32* p_true_nodes_owner = new VariableNodeInt32(VariableBuildInfo(mesh,"TrueNodesOwner"));
    m_part_indexes[i].m_true_cells_owner = p_true_cells_owner;
    m_part_indexes[i].m_true_nodes_owner = p_true_nodes_owner;
    VariableCellInt32& true_cells_owner = *p_true_cells_owner;      
    VariableNodeInt32& true_nodes_owner = *p_true_nodes_owner;      
    IItemFamily* current_cell_family = mesh->cellFamily();
    IItemFamily* current_node_family = mesh->nodeFamily();
    VariableItemInt32& cells_new_owner(current_cell_family->itemsNewOwner());
    VariableItemInt32& nodes_new_owner(current_node_family->itemsNewOwner());
    bool is_dynamic = mesh->isDynamic();
    mesh->modifier()->setDynamic(true);
    // Premier partitionnement (optionnel) pour donner un premier resultat correct
    //mesh_partitioner->partitionMesh(mesh);
    //mesh->exchangeItems(false);
    
    // Partitionnement final
    {
      sd->timeStats()->dumpTimeAndMemoryUsage(pm);
      Timer t(sd,"InitPartTimer",Timer::TimerReal);
      {
        Timer::Sentry ts(&t);
        mesh_partitioner->partitionMesh(mesh,nb_part);
      }
      tm->info() << "Partitioning time t=" << t.lastActivationTime();
      sd->timeStats()->dumpTimeAndMemoryUsage(pm);
    }
    ENUMERATE_CELL(icell,current_cell_family->allItems()){
      Int32 new_owner = cells_new_owner[icell];
      true_cells_owner[icell] = new_owner;
      cells_new_owner[icell] = new_owner % nb_rank;
    }
    ENUMERATE_NODE(inode,current_node_family->allItems()){
      true_nodes_owner[inode] = nodes_new_owner[inode];
    }
    _printStats(nb_part,mesh,true_cells_owner);
    mesh->utilities()->changeOwnersFromCells();
    //mesh->modifier()->setDynamic(true);
    //PRIMARYMESH_CAST(mesh)->exchangeItems();
    bool compact = mesh->properties()->getBool("compact");
    mesh->properties()->setBool("compact", true);
    mesh->toPrimaryMesh()->exchangeItems();
    mesh->modifier()->setDynamic(is_dynamic);
    mesh->properties()->setBool("compact", compact);
  }

  // ajout d'une 2ème couche de mailles 
  // il ne faut plus faire exchangeItems avec les 2 couches de mailles
  IMesh* mesh = meshes[0];
  if (m_main->options()->nbGhostLayer()==2)
    mesh->updateGhostLayers(false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneInitialPartitioner::
_printStats(Integer nb_part,IMesh* mesh,VariableCellInt32& new_owners)
{
  Int64UniqueArray nb_cells(nb_part,0);
  ENUMERATE_CELL(icell,mesh->ownCells()){
    Int32 new_owner = new_owners[icell];
    ++nb_cells[new_owner];
  }
  IParallelMng* pm = mesh->parallelMng();
  pm->reduce(Parallel::ReduceSum,nb_cells);
  ITraceMng* tm = m_sub_domain->traceMng();
  tm->info() << " -- Partitioning statistics --";
  tm->info() << "   Part              NbCell";
  for( Integer i=0; i<nb_part; ++i ){
    tm->info() << Trace::Width(6) << i << Trace::Width(18) << nb_cells[i];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArcaneCasePartitioner::
ArcaneCasePartitioner(const ServiceBuildInfo& sb)
: ArcaneArcaneCasePartitionerObject(sb)
{
  m_init_part = new ArcaneInitialPartitioner(this,sb.subDomain());
  info() << "** ** SET INITIAL PARTITIONER 2";
  sb.subDomain()->setInitialPartitioner(m_init_part);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArcaneCasePartitioner::
~ArcaneCasePartitioner()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneCasePartitioner::
execute()
{
  Int32 nb_part = options()->nbCutPart();
  info() << "ArcaneCasePartitioner::execute() nb_part=" << nb_part;
  if (nb_part!=0){
    subDomain()->timeStats()->dumpTimeAndMemoryUsage(subDomain()->parallelMng());
    _partitionMesh(nb_part);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneCasePartitioner::
_partitionMesh(Int32 nb_part)
{
  //String lib_name = m_main->options()->library(); //"Metis";
  ISubDomain* sd = subDomain();
  IMesh* current_mesh = mesh();
  IParallelMng* pm = sd->parallelMng();
  Int32 nb_rank = pm->commSize();
  Int32 my_rank = pm->commRank();

  //FactoryT<IMeshWriter> mesh_writer_factory(sd->serviceMng());
  String mesh_writer_name = options()->writerServiceName();
  if (mesh_writer_name.empty())
    pfatal() << "No service selected to write the mesh";
  ServiceBuilder<IMeshWriter> sb(sd);
  auto mesh_writer = sb.createReference(mesh_writer_name,SB_Collective);

  String pattern = options()->meshFileNamePattern();
  info() << "Mesh file pattern=" << pattern;

  // Partitionne le maillage.
  // En retour, \a cells_new_owner contient le numéro de la partie à laquelle
  // chaque maille appartiendra. Pour sauver le fichier, il faut que toutes
  // les mailles d'une partie soient sur le même sous-domaine. Pour cela,
  // on stocke le numéro de la partie dans \a true_cells_owner, puis
  // on échange le maillage.
  //mesh_partitioner->partitionMesh(current_mesh,nb_part);
  VariableCellInt32 true_cells_owner(*m_init_part->m_part_indexes[0].m_true_cells_owner);
  VariableNodeInt32 true_nodes_owner(*m_init_part->m_part_indexes[0].m_true_nodes_owner);
  IItemFamily* current_cell_family = mesh()->cellFamily();
  //VariableItemInt32& cells_new_owner(current_cell_family->itemsNewOwner());
  CellGroup current_all_cells = current_cell_family->allItems();
  Integer total_current_nb_cell = pm->reduce(Parallel::ReduceSum,current_all_cells.own().size());
  info() << "TOTAL_NB_CELL=" << total_current_nb_cell;

  IMainFactory* main_factory = sd->application()->mainFactory();
  IPrimaryMesh* new_mesh = main_factory->createMesh(sd,pm->sequentialParallelMng(),"SubMesh");
  new_mesh->setDimension(mesh()->dimension());
  // Pour optimiser, il n'y a pas besoin de trier ni de compacter les entités.
  new_mesh->properties()->setBool("compact",false);
  new_mesh->properties()->setBool("sort",false);
  new_mesh->modifier()->setDynamic(true);
  new_mesh->allocateCells(0,Int64ConstArrayView(),true);

  Integer saved_nb_cell = 0;
  Integer min_nb_cell = total_current_nb_cell;
  Integer max_nb_cell = 0;
  
  if (options()->createCorrespondances())
    _initCorrespondance(my_rank);

  // recherche une fois pour toute les id max
  Integer maxLocalIdCell = mesh()->cellFamily()->maxLocalId();
  Integer maxLocalIdNode = mesh()->nodeFamily()->maxLocalId();

  // Force le propriétaire des entités pour au sous-domaine 0 car
  // new_mesh utilise un parallelMng() séquentiel.
  for( IItemFamily* family : mesh()->itemFamilies() ){
    ENUMERATE_ITEM(iitem,family->allItems()){
      iitem->mutableItemBase().setOwner(0,0);
    }
  }

  // Pour chaque partie à traiter, créé un maillage
  // contenant les entités de cette partie
  // et le sauvegarde
  info() << "NbPart=" << nb_part << " my_rank=" << my_rank;
  for( Integer i=0; i<nb_part; ++i ){
    if ((i % nb_rank)!=my_rank){
      if (my_rank==0 && options()->createCorrespondances()){
	
        info()<<"Receive on master to build correspondence file on sub-domain "<<i
              <<" sent from processor "<<i % nb_rank;
        Int32UniqueArray taillesTab(2);
        Int64UniqueArray nodesUniqueId;
        Int64UniqueArray cellsUniqueId;
	
        pm->recv(taillesTab, i % nb_rank);
        nodesUniqueId.resize(taillesTab[0]);
        cellsUniqueId.resize(taillesTab[1]);
        pm->recv(nodesUniqueId, i % nb_rank);
        pm->recv(cellsUniqueId, i % nb_rank);
        _writeCorrespondance(i, nodesUniqueId, cellsUniqueId);
      }
      continue;
    }

    new_mesh->destroyGroups();
    new_mesh->modifier()->clearItems();
    new_mesh->modifier()->endUpdate();
    UniqueArray<Cell> cells_selected_for_new_mesh;
    ENUMERATE_CELL(icell,current_all_cells.own()){
      if (true_cells_owner[icell]==i){
        Cell cell = *icell;
        cells_selected_for_new_mesh.add(cell);
        //info() << "ADD CELL " << ItemPrinter(cell);
      }
    }

    // sélectionne les mailles fantômes en plus si nécessaire
    _addGhostLayers(current_all_cells,  cells_selected_for_new_mesh, options()->nbGhostLayer(), maxLocalIdCell, maxLocalIdNode);

    Int32UniqueArray cells_local_id;
    Int64UniqueArray cells_unique_id;
    for( Integer j=0, js=cells_selected_for_new_mesh.size(); j<js; ++j ){
      Cell cell = cells_selected_for_new_mesh[j];
      cells_local_id.add(cell.localId());
      cells_unique_id.add(static_cast<Int64>(cell.uniqueId()));
    }

    Integer nb_cell_to_copy = cells_local_id.size();
    SerializeBuffer buffer;
    current_mesh->serializeCells(&buffer,cells_local_id);
    info() << "NB_CELL_TO_SERIALIZE=" << nb_cell_to_copy;
    new_mesh->modifier()->addCells(&buffer);
    new_mesh->modifier()->endUpdate();
    // Pour mettre a jour les coordonnees
    //new_mesh->nodeFamily()->endUpdate();
    ItemInternalList new_cells = new_mesh->itemsInternal(IK_Cell);
    ItemInternalList current_cells = current_mesh->itemsInternal(IK_Cell);
    VariableNodeReal3& new_coordinates(new_mesh->nodesCoordinates());
    VariableNodeReal3& current_coordinates(current_mesh->toPrimaryMesh()->nodesCoordinates());
    Int32UniqueArray new_cells_local_id(nb_cell_to_copy);
    new_mesh->cellFamily()->itemsUniqueIdToLocalId(new_cells_local_id,cells_unique_id);
    for( Integer zid=0; zid<nb_cell_to_copy; ++zid ){
      Cell current_cell = current_cells[cells_local_id[zid]];
      Cell new_cell = new_cells[new_cells_local_id[zid]];
      if (current_cell.uniqueId()!=new_cell.uniqueId())
        fatal() << "Inconsistent unique ids";
      Integer nb_node = current_cell.nbNode();
      //info() << "Current=" << ItemPrinter(current_cell)
      //       << " new=" << ItemPrinter(new_cell)
      //       << " nb_node=" << nb_node;
      for( Integer z2=0; z2<nb_node; ++z2 ){
        Real3 coord = current_coordinates[current_cell.node(z2)];
	//         info() << "Node=" << ItemPrinter(new_cell.node(z2)) << " coord=" << coord
	//                << " orig_node=" << ItemPrinter(current_cell.node(z2));
        new_coordinates[new_cell.node(z2)] = coord;
        // Positionne le propriétaire final du noeud
        new_cell.node(z2).mutableItemBase().setOwner(true_nodes_owner[current_cell.node(z2)],0);
      }
    }
    // Maintenant, il faut recopier les groupes
    {
      _computeGroups(current_mesh->nodeFamily(),new_mesh->nodeFamily());
      _computeGroups(current_mesh->edgeFamily(),new_mesh->edgeFamily());
      _computeGroups(current_mesh->faceFamily(),new_mesh->faceFamily());
      _computeGroups(current_mesh->cellFamily(),new_mesh->cellFamily());

      if (options()->nbGhostLayer()>0)
        _addGhostGroups(new_mesh, cells_selected_for_new_mesh, true_cells_owner, true_nodes_owner, new_cells_local_id, i);
    }
    Integer new_nb_cell = new_mesh->nbCell();
    info() << "NB_NEW_CELL=" << new_nb_cell;
    min_nb_cell = math::min(min_nb_cell,new_nb_cell);
    max_nb_cell = math::max(max_nb_cell,new_nb_cell);
    saved_nb_cell += new_nb_cell;
    String filename;
    if (pattern.empty()){
      StringBuilder sfilename = "cut_mesh_";
      sfilename += i;
      sfilename += ".mli2";
      filename = sfilename;
    }
    else{
      //ATTENTION potentiel debordement si pattern est trop long.
      //Verifier aussi qu'il y a un %d. A terme, utiliser String::format()
      char buf[4096];
      if (pattern.length()>128){
        pfatal() << "Pattern too long (max=128)";
      }
      sprintf(buf,pattern.localstr(),i);
      filename = String(StringView(buf));
    }
    {
      info() << "Writing mesh file filename='" << filename << "'";
      bool is_bad = mesh_writer->writeMeshToFile(new_mesh, filename);
      if (is_bad)
        ARCANE_FATAL("Can not write mesh file '{0}'", filename);
    }

    // Fichier Correspondance
    if (options()->createCorrespondances()){
      info()<<"Participation to build correspondence file on sub-domain "<<i;
      
      Int32UniqueArray taillesTab;
      taillesTab.add(new_mesh->nodeFamily()->nbItem());
      taillesTab.add(new_mesh->cellFamily()->nbItem());
      Int64UniqueArray nodesUniqueId(taillesTab[0]);
      Int64UniqueArray cellsUniqueId(taillesTab[1]);

      NodeInfoListView nodes(new_mesh->nodeFamily());
      for( int j=0; j<taillesTab[0]; ++j ){
        Node node = nodes[j];
        nodesUniqueId[j] = node.uniqueId();
      }

      CellInfoListView cells(new_mesh->cellFamily());
      for( int j=0; j<taillesTab[1]; ++j ){
        Cell cell = cells[j];
        cellsUniqueId[j] = cell.uniqueId();
      }

      if (my_rank!=0){
        pm->send(taillesTab, 0);
        pm->send(nodesUniqueId, 0);
        pm->send(cellsUniqueId, 0);
      }
      else {
        _writeCorrespondance(i, nodesUniqueId, cellsUniqueId);
      }
    }
  } // end i<nb_part

  Integer total_new_nb_cell = pm->reduce(Parallel::ReduceSum,saved_nb_cell);
  Integer total_min_nb_cell = pm->reduce(Parallel::ReduceMin,min_nb_cell);
  Integer total_max_nb_cell = pm->reduce(Parallel::ReduceMax,max_nb_cell);
  info() << "TOTAL_NEW_NB_CELL=" << total_new_nb_cell
         << " min=" << total_min_nb_cell
         << " max=" << total_max_nb_cell
         << " computed_average=" << (total_current_nb_cell/nb_part);

  subDomain()->timeStats()->dumpTimeAndMemoryUsage(pm);

  if (options()->createCorrespondances())
    _finalizeCorrespondance(my_rank);

  if (options()->nbGhostLayer()==0)
    if (total_new_nb_cell!=total_current_nb_cell)
      pfatal() << "Bad number of saved cells current=" << total_current_nb_cell
               << " saved=" << total_new_nb_cell;

  pinfo()<<"Total Memory Used : "<<platform::getMemoryUsed();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneCasePartitioner::
_initCorrespondance(Int32 my_rank)
{
  info()<<" _initCorrespondance("<<my_rank<<")";

  if (my_rank)
    return;

  m_sortiesCorrespondance.open("Correspondances");

  if (m_sortiesCorrespondance.fail ()){
    pfatal() << "Unable to write to file 'Correspondances' ";
  }

  m_sortiesCorrespondance << "<?xml version=\"1.0\" encoding=\"ISO-8859-1\" ?>\n";
  m_sortiesCorrespondance << "<!-- Correspondance file generated by Arcane/Decoupe3D V2 -->\n";
  m_sortiesCorrespondance << "\n<cpus>\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneCasePartitioner::
_writeCorrespondance(Int32 rank, Int64Array& nodesUniqueId, Int64Array& cellsUniqueId)
{
  info()<<" _writeCorrespondance("<<rank<<", nodesUniqueId.size() = "
	<<nodesUniqueId.size()<<", cellsUniqueId.size() = "<<cellsUniqueId.size()<<")";

  m_sortiesCorrespondance << "  <cpu id=\"" << rank << "\">" << "\n"
			  << "    <noeuds>" << "\n" << "    ";
  for( Integer i=0; i<nodesUniqueId.size(); ++i )
    m_sortiesCorrespondance <<nodesUniqueId[i]<< " ";
  
  m_sortiesCorrespondance << "\n" << "    </noeuds>"
			  << "\n"
			  << "    <mailles>" << "\n"
			  << "      ";
  for( Integer i=0; i<cellsUniqueId.size(); ++i )
    m_sortiesCorrespondance <<cellsUniqueId[i]<< " ";
  m_sortiesCorrespondance << "\n" << "    </mailles>" << "\n"
			  << "  </cpu>" << "\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneCasePartitioner::
_finalizeCorrespondance(Int32 my_rank)
{
  if (my_rank)
    return;

  m_sortiesCorrespondance << "</cpus>\n";
  m_sortiesCorrespondance.close ();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Recopie les groupes de la famille courante dans la nouvelle.
 *
 * Le principe est le suivant:
 * 1. pour chaque genre d'entité, détermine la liste des localId()
 * des entités de ce genre dans le maillage d'origine.
 * 2. A partir de cette liste, construit un tableau indiquant
 * pour chaque localId() du maillage d'origine son localId() dans le nouveau
 * maillage (ou NULL_ITEM_LOCAL_ID si l'entité est absente).
 * 3. Parcours les groupes d'origine et construit
 * pour chacun la liste des entités à ajouter au nouveau maillage.
 */
void ArcaneCasePartitioner::
_computeGroups(IItemFamily* current_family,IItemFamily* new_family)
{
  info() << "Compute groups family=" << current_family->name();

  ItemGroup new_all_items = new_family->allItems();
  Integer nb_new_item = new_all_items.size();

  Int64UniqueArray new_items_uid(nb_new_item);
  Int32UniqueArray new_items_lid(nb_new_item);
  {
    Integer index = 0;
    ENUMERATE_ITEM(iitem,new_all_items){
      new_items_uid[index] = (*iitem).uniqueId();
      new_items_lid[index] = iitem.itemLocalId();
      ++index;
    }
  }
  Int32UniqueArray items_lid(nb_new_item);
  // Détermine le localId() dans le maillage d'origine des entités
  current_family->itemsUniqueIdToLocalId(items_lid,new_items_uid);

  Int32UniqueArray items_current_to_new_local_id(current_family->maxLocalId());
  items_current_to_new_local_id.fill(NULL_ITEM_LOCAL_ID);
  for( Integer i=0; i<nb_new_item; ++i )
    items_current_to_new_local_id[items_lid[i]] = new_items_lid[i];

  Int32UniqueArray create_local_ids;
  for( ItemGroupCollection::Enumerator igroup(current_family->groups()); ++igroup; ){
    ItemGroup group = *igroup;
    if (group.isOwn())
      continue;
    if (group.isAllItems())
      continue;
    create_local_ids.clear();
    ENUMERATE_ITEM(iitem,group){
      Int32 current_uid = iitem.itemLocalId();
      Int32 new_lid = items_current_to_new_local_id[current_uid];
      if (new_lid!=NULL_ITEM_LOCAL_ID)
        create_local_ids.add(new_lid);
    }
    new_family->createGroup(group.name(),create_local_ids,true);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/* \brief Ajoute au tableau de mailles le nombre de couches de mailles désiré
 */
void ArcaneCasePartitioner::
_addGhostLayers(CellGroup current_all_cells, Array<Cell>& cells_selected_for_new_mesh,
                Integer nbCouches,Integer maxLocalIdCell, Integer maxLocalIdNode)
{
  if (nbCouches==0)
    return;

  Int32UniqueArray filtre_lid_cell(maxLocalIdCell);
  filtre_lid_cell.fill(0);
  Int32UniqueArray filtre_lid_node(maxLocalIdNode);
  filtre_lid_node.fill(0);

  // on marque les mailles déjà sélectionnées
  for( Integer j=0, js=cells_selected_for_new_mesh.size(); j<js; ++j ){
    Cell cell = cells_selected_for_new_mesh[j];
    filtre_lid_cell[cell.localId()] = 1;
  }

  // recherhe pour tous les noeuds associés aux mailles sélectionnées une mailles reliée
  // à ce même noeud qui ne soit pas sélectionnée
  for( Integer j=0, js=cells_selected_for_new_mesh.size(); j<js; ++j ){
    Cell cell = cells_selected_for_new_mesh[j];

    NodeVectorView nodes = cell.nodes();
    for( Integer k=0, ks=nodes.size(); k<ks; ++k){
      Node node = nodes[k];
      if (filtre_lid_node[node.localId()]==0){
        // les mailles reliées par un noeud
        CellVectorView cells_vois = node.cells();

        for( Integer i=0, is=cells_vois.size(); i<is; ++i ){
          Cell cell_vois = cells_vois[i];
          if (filtre_lid_cell[cell_vois.localId()]==0){

            // ajoute la maille qui n'a pas encore été vue	    
            cells_selected_for_new_mesh.add(cell_vois);

            filtre_lid_cell[cell_vois.localId()] = 1;
          }
        }
        filtre_lid_node[node.localId()] = 1;
      }
    }
  }  

  // pour la deuxième couche (si besoin) il est plus simple de le faire récurcivement
  _addGhostLayers(current_all_cells, cells_selected_for_new_mesh,  nbCouches-1, maxLocalIdCell,  maxLocalIdNode);  

}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/* \brief Ajoute les groupes de mailles TOUT, LOCAL et MF_* en fonctions des groupes voisins
 *        Ajoute aussi le groupe de noeuds LOCALN (mais pas les NF_*)
 */
void ArcaneCasePartitioner::
_addGhostGroups(IMesh* new_mesh, Array<Cell>& cells_selected_for_new_mesh, VariableCellInt32& true_cells_owner,
                VariableNodeInt32& true_nodes_owner,
                Int32Array& new_cells_local_id, Integer id_loc)
{
  info()<<"ArcaneCasePartitioner::_addGhostGroups (id_loc = "<<id_loc<<")";
  // il faut déterminer les groupes voisins existant
  // on utilise un "map" pour stocker les différents sous-domaines qui apparaissent et le nombre de mailles dedans
  std::map<Integer, Integer> dom_vois;
  for( Integer j=0, js=cells_selected_for_new_mesh.size(); j<js; ++j ){
    Cell cell = cells_selected_for_new_mesh[j];
    dom_vois[true_cells_owner[cell]] += 1;
  }
  
  // on utilise une seconde map pour lister les mailles suivant le domaine de destination
  std::map<Integer,SharedArray<Int32> > map_groupes;
  for (std::map<Integer, Integer>::const_iterator iter=dom_vois.begin(); iter!=dom_vois.end(); ++iter){
    Integer no_sous_dom = iter->first;
    Integer nb_mailles_sous_dom = iter->second;

    // réservation de la mémoire pour les différentes listes
    Int32Array& tab = map_groupes[no_sous_dom];
    tab.reserve(nb_mailles_sous_dom);
  }

  for( Integer j=0, js=cells_selected_for_new_mesh.size(); j<js; ++j ){
    Cell cell = cells_selected_for_new_mesh[j];
    Integer no_sous_dom = true_cells_owner[cell];
   
    // remplissage des listes par sous-domaine
    Int32Array & liste_lid = map_groupes[no_sous_dom];
    liste_lid.add(new_cells_local_id[j]);
  }

  // création (si nécessaire) des différents groupes et on y met les mailles
  for (std::map<Integer,SharedArray<Int32> >::iterator iter=map_groupes.begin(); iter!=map_groupes.end(); ++iter){
    Integer no_sous_dom = iter->first;
    Int32Array & liste_lid = iter->second;

    ItemGroup groupe_loc;
    if (no_sous_dom==id_loc)
      groupe_loc = new_mesh->cellFamily()->findGroup("LOCAL", true);
    else {
      String nom_mf("MF_");
      nom_mf = nom_mf+no_sous_dom;
      groupe_loc = new_mesh->cellFamily()->findGroup(nom_mf, true);
    }
    
    groupe_loc.addItems(liste_lid, false);
  }

  // Faire le groupe LOCALN : noeuds locaux
  {
    // TODO: Optimiser la maniere de construire ce groupe
    Int32UniqueArray liste_lid;
    Integer nbnodes = new_mesh->nodeFamily()->nbItem();
    liste_lid.reserve(nbnodes);
    NodeInfoListView nodes(new_mesh->nodeFamily());
    for (int j= 0 ; j < nbnodes ; ++j) {
      Node node= nodes[j];
      if (true_nodes_owner[node] == id_loc)
        liste_lid.add(node.localId());
    }

    ItemGroup groupe_loc = new_mesh->nodeFamily()->findGroup("LOCALN", true);
    groupe_loc.addItems(liste_lid, false);
  }


  // le groupe avec toute les mailles
  ItemGroup groupe_glob = new_mesh->cellFamily()->findGroup("TOUT", true);
  
  groupe_glob.addItems(new_cells_local_id, false);
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_ARCANECASEPARTITIONER(ArcaneCasePartitioner,ArcaneCasePartitioner);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
