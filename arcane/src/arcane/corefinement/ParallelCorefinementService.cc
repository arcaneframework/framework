// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#ifdef WIN32
#include <iso646.h>
#endif

#include "arcane/corefinement/ParallelCorefinementService.h"

#include <arcane/utils/List.h>
#include <arcane/utils/Limits.h>
#include <arcane/utils/ITraceMng.h>
#include <arcane/utils/ScopedPtr.h>

#include <arcane/core/IItemFamily.h>
#include <arcane/core/IMesh.h>
#include <arcane/core/IMeshModifier.h>
#include <arcane/core/IMeshSubMeshTransition.h>
#include <arcane/core/IParallelMng.h>
#include <arcane/core/ISerializeMessageList.h>
#include <arcane/core/Timer.h>
#include <arcane/core/internal/SerializeMessage.h>

#include <arcane/core/IParallelExchanger.h>

#include <set>
#include <list>
#include <map>
#include <vector>

#include <arcane/geometry/IGeometry.h>
#include <arcane/geometry/IGeometryMng.h>
#include <arcane/corefinement/surfaceutils/ISurface.h>

using namespace Arcane;
using namespace Arcane::Numerics;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ParallelCorefinementService::Internal
{
 public:

  Internal(ParallelCorefinementService* p)
  : migrationTimer(p->subDomain(), "CorefinementMigration", Timer::TimerVirtual)
  , corefinementTimer(p->subDomain(), "CorefinementAll", Timer::TimerVirtual)
  {
    ;
  }

 public:

  class Box
  {
   public:

    Box(const Real3& lower_left,
        const Real3& upper_right,
        const VariableNodeReal3& nodesCoordinates)
    : m_lower_left(lower_left)
    , m_upper_right(upper_right)
    , m_nodes_coordinates(nodesCoordinates)
    {}
    bool isInside(const Real3& coord) const
    {
      return coord.x > m_lower_left.x and coord.y > m_lower_left.y and coord.z > m_lower_left.z and coord.x < m_upper_right.x and coord.y < m_upper_right.y and coord.z < m_upper_right.z;
    }

    bool isInside(const Face& face)
    {
      for (Node node : face.nodes()) {
        // m_trace->info() << inode->localId() << " " << m_nodes_coordinates[inode] << " : " << isInside(m_nodes_coordinates[inode]);
        if (isInside(m_nodes_coordinates[node]))
          return true;
      }
      return false;
    }

    ITraceMng* m_trace;

   private:

    const Real3 m_lower_left, m_upper_right;
    const VariableNodeReal3& m_nodes_coordinates;
  };

  //! test de distance entre faces
  /*! Compatible avec Box::isInside */
  class CheckCloseFaces {
  public:
    CheckCloseFaces(const VariableNodeReal3 & nodesCoordinates, const Real distance) 
      : m_nodes_coordinates(nodesCoordinates),
        m_distance(distance) { }
  
    bool operator()(const Face & faceA, const Face & faceB)
    {
      for (Node nodeA : faceA.nodes()) {
        for (Node nodeB : faceB.nodes()) {
          if (math::normeR3(m_nodes_coordinates[nodeA] - m_nodes_coordinates[nodeB]) < m_distance)
            return true;
        }
      }
      return false;
    }

   private:
    const VariableNodeReal3 & m_nodes_coordinates;
    const Real m_distance;
  };

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  struct NodeComparator {
    typedef std::pair<Integer,Integer> IntPair;
    bool operator()(const IntPair & a, const IntPair & b) const {
      if (a.first == b.first)
        return a.second < b.second;
      else
        return a.first < b.first;
    }
    static IntPair order(const IntPair & p) {
      if (p.first > p.second)
        return IntPair(p.second,p.first);
      else
        return p;
    }
  };

  static void surfaceSetup(std::map<NodeComparator::IntPair, Integer > & edges, FaceGroup group, IGeometry * geometry) {
    typedef NodeComparator::IntPair IntPair;
    Real3 normal(0.,0.,0.);
    ENUMERATE_FACE(iface,group) {
      const Face face = *iface;
      const NodeVectorView nodes = face.nodes();
      const Integer nnodes = nodes.size();
      for(Integer i=0;i<nnodes;++i) {
        const IntPair p = NodeComparator::order(IntPair(nodes[i].localId(),nodes[(i+1)%nnodes].localId()));
        edges[p]++;
      }
    }
  }

  class AddEdges {
  public:
    void operator()(const NodeComparator::IntPair & p) {
      const Integer indexA = mapNodes[p.first];
      const Integer indexB = mapNodes[p.second];
      if (not indexA and not indexB) {
        mapNodes[p.first] = mapNodes[p.second] = newRef();
      } else if (not indexA and indexB) {
        mapNodes[p.first] = mapNodes[p.second];
      } else if (indexA and not indexB) {
        mapNodes[p.second] = mapNodes[p.first];
      } else if (indexA and indexB) {
        allRefs[mapNodes[p.second]-1] = allRefs[mapNodes[p.first]-1];
      }
    }
  
    Integer connexCount() {
      std::set<Integer> refSet;
      for(Integer i=0;i<(Integer)allRefs.size();++i)
        refSet.insert(*allRefs[i]);
      return refSet.size();
    }
  
  private:
    std::vector<std::shared_ptr<Integer> > allRefs;
    typedef std::map<Integer,Integer> MapNodes;
    MapNodes mapNodes;
  
    Integer newRef() {
      Integer ref = allRefs.size()+1; // reserve 0 pour la non-valeur
      allRefs.push_back(std::shared_ptr<Integer>(new Integer(ref)));
      return ref;
    }
  };


  static void exchange_items(IMesh * mesh,
                             UniqueArray<std::set<Int32> > & nodes_to_send,
                             UniqueArray<std::set<Int32> > & faces_to_send,
                             UniqueArray<std::set<Int32> > & cells_to_send) {
    ITraceMng * traceMng = mesh->traceMng();  
    if (mesh == NULL) traceMng->fatal() << "Incompatible Mesh type";
    IParallelMng * parallel_mng = mesh->parallelMng();

    ScopedPtrT<IParallelExchanger> exchanger(parallel_mng->createExchanger());
    
    const Integer nbSubDomain = parallel_mng->commSize();  

    // Initialisation de l'échangeur de données
    for(Integer isd=0;isd<nbSubDomain;++isd)
      if (not cells_to_send[isd].empty())
        exchanger->addSender(isd);
    exchanger->initializeCommunicationsMessages();

    for(Integer i=0, ns=exchanger->nbSender(); i<ns; ++i) 
      {
        ISerializeMessage* sm = exchanger->messageToSend(i);
        Int32 rank = sm->destRank();
        ISerializer* s = sm->serializer();
        const std::set<Int32> & cell_set = cells_to_send[rank];
        Int32UniqueArray items_to_send(cell_set.size());
        std::copy(cell_set.begin(), cell_set.end(), items_to_send.begin());
        mesh->/*modifier()->*/serializeCells(s, items_to_send);
      }
    exchanger->processExchange();
    
    for( Integer i=0, ns=exchanger->nbReceiver(); i<ns; ++i )
      {
        ISerializeMessage* sm = exchanger->messageToReceive(i);
        ISerializer* s = sm->serializer();
            mesh->modifier()->addCells(s);
      }
    
    mesh->modifier()->endUpdate(true, false); // recalcul des ghosts sans suppression des anciens
  }

  Timer migrationTimer;
  Timer corefinementTimer;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParallelCorefinementService::
ParallelCorefinementService(const Arcane::ServiceBuildInfo & sbi) : 
  ArcaneParallelCorefinementObject(sbi),
  m_internal(new Internal(this))
{
  ;
}

/*---------------------------------------------------------------------------*/

ParallelCorefinementService::
~ParallelCorefinementService() 
{
  delete m_internal;
}

/*---------------------------------------------------------------------------*/

void 
ParallelCorefinementService::
init(const FaceGroup & masterGroup, 
     const FaceGroup & slaveGroup, 
     const Real boxTolerance,
     IGeometry * geometry)
{
  m_master_group = masterGroup;
  m_slave_group = slaveGroup;
  m_box_tolerance = boxTolerance;
  m_geometry = geometry;
}

/*---------------------------------------------------------------------------*/

void 
ParallelCorefinementService::
update()
{
//   info() << "All groups list : ";
//   ItemGroupCollection coll = mesh()->groups();
//   for(ItemGroupCollection::Iterator i=coll.begin(); i !=coll.end(); ++i) {
//     ItemGroup g = *i;
//     info() << "\tGroup : " << g.name() << " size=" << g.size();
//   }
  m_internal->corefinementTimer.start();
  m_internal->migrationTimer.start();

  pinfo() << "Surface infos berfore migration " 
          << m_master_group.name() << "(" << m_master_group.size() << ") and "
          << m_slave_group.name() << "(" << m_slave_group.size() << ")";

  IParallelMng * parallel_mng = subDomain()->parallelMng();

  // 1- Calcul des boites englobantes coté Master
  //    + Slave pour minimiser (et non eviter) de multiples composantes connexes 
  //    (non supporté par le coraffinement simple)
  const Real maxValue = FloatInfo<Real>::maxValue();
  const VariableNodeReal3 & nodesCoordinates = PRIMARYMESH_CAST(mesh())->nodesCoordinates();
  Real3 
    master_lower_left(maxValue,maxValue,maxValue), 
    master_upper_right(-maxValue,-maxValue,-maxValue),
    slave_lower_left(maxValue,maxValue,maxValue), 
    slave_upper_right(-maxValue,-maxValue,-maxValue);
    
  
  // Not optimized box computation (many nodes are min/max several times)
  // Master BOX
  ENUMERATE_FACE(iface,m_master_group) {
    for (Node node : iface->nodes()) {
            master_lower_left = math::min(master_lower_left, nodesCoordinates[node]);
            master_upper_right = math::max(master_upper_right, nodesCoordinates[node]);
    }
  }
  ENUMERATE_FACE(iface,m_slave_group) {
    for (Node node : iface->nodes()) {
            master_lower_left = math::min(master_lower_left, nodesCoordinates[node]);
            master_upper_right = math::max(master_upper_right, nodesCoordinates[node]);
    }
  }

  // Slave BOX
  ENUMERATE_FACE(iface,m_slave_group) {
    for (Node node : iface->nodes()) {
            slave_lower_left = math::min(slave_lower_left, nodesCoordinates[node]);
            slave_upper_right = math::max(slave_upper_right, nodesCoordinates[node]);
    }
  }

  // Extend current box
  master_lower_left.subSame(m_box_tolerance);
  master_upper_right.addSame(m_box_tolerance);
  slave_lower_left.subSame(m_box_tolerance);
  slave_upper_right.addSame(m_box_tolerance);
  debug(Trace::Medium) << "Local Box sizes :"
                       << " M= " << master_lower_left << " - " << master_upper_right
                       << " S= " << slave_lower_left << " - " << slave_upper_right;
  
  // 2- migration des faces esclaves en fonction des boites maitres
  //  + migration des faces maitres 'utiles' vers les processeurs esclaves
  UniqueArray<Real3> myBox;
  myBox.add(master_lower_left);
  myBox.add(master_upper_right);
  myBox.add(slave_lower_left);
  myBox.add(slave_upper_right);

  const Integer nbSubDomain = subDomain()->nbSubDomain();
  UniqueArray<Real3> allBoxes(4*nbSubDomain);
  parallel_mng->allGather(myBox,allBoxes);

  // NB: ne traite pas les aretes.
  // exchangers utilisés en mode slave->master puis master->slave
  UniqueArray<std::set<Int32> > nodes_to_send; nodes_to_send.resize(nbSubDomain);
  UniqueArray<std::set<Int32> > faces_to_send; faces_to_send.resize(nbSubDomain);
  UniqueArray<std::set<Int32> > cells_to_send; cells_to_send.resize(nbSubDomain);

  // On peut optimiser les calculs d'inclusion en ayant déjà la boite englobante locale
  // Si pas d'intersection des boites locale et distante => rien à faire.
#ifndef NO_USER_WARNING
#warning "optimisations possibles"
#endif /* NO_USER_WARNING */
  for(Integer i_sub_domain=0; i_sub_domain<nbSubDomain;++i_sub_domain) {
    if (i_sub_domain == subDomain()->subDomainId()) continue;
    Internal::Box masterBox(allBoxes[4*i_sub_domain],allBoxes[4*i_sub_domain+1],nodesCoordinates);
    masterBox.m_trace = traceMng();
    ENUMERATE_FACE(iface,m_slave_group) {
      if (masterBox.isInside(*iface)) {
        faces_to_send[i_sub_domain].insert(iface->localId());
        Cell boundaryCell = iface->boundaryCell();
        if (boundaryCell.null())
          fatal() << "Non boundary face used in co-refinement";
        cells_to_send[i_sub_domain].insert(boundaryCell.localId());
        for (Node node : boundaryCell.nodes())
          nodes_to_send[i_sub_domain].insert(node.localId());
            }
    }

    Internal::Box slaveBox(allBoxes[4*i_sub_domain+2],allBoxes[4*i_sub_domain+3],nodesCoordinates);
    slaveBox.m_trace = traceMng();
    ENUMERATE_FACE(iface,m_master_group) {
      if (masterBox.isInside(*iface)) {
        faces_to_send[i_sub_domain].insert(iface->localId());
        Cell boundaryCell = iface->boundaryCell();
        if (boundaryCell.null())
          fatal() << "Non boundary face used in co-refinement";
        cells_to_send[i_sub_domain].insert(boundaryCell.localId());
        for (Node node : boundaryCell.nodes())
          nodes_to_send[i_sub_domain].insert(node.localId());
            }
    }
  }

  Internal::exchange_items(mesh(),nodes_to_send,faces_to_send,cells_to_send);

  debug(Trace::Medium) << "Surface infos after migration " 
                       << m_master_group.name() << "(" << m_master_group.size() << ") and "
                       << m_slave_group.name() << "(" << m_slave_group.size() << ")";

  m_internal->migrationTimer.stop();

  // 3- Calcul des composantes connexes
  // On vérifie pour l'instant que localement nous n'avons qu'une composante connexe.
  // Pour cela, on vérifie que le bord des surfaces ne forment qu'une courbe
    
  // Choix des surfaces pour le co-raffinement local
  FaceGroup slave_coref_group = m_slave_group;
  FaceGroup master_coref_group = m_master_group;

  typedef Internal::NodeComparator::IntPair IntPair;
  std::map<IntPair, Integer > slave_edges; // paires des sommets indexés par leurs localIds
  std::map<IntPair, Integer > master_edges; // paires des sommets indexés par leurs localIds
  Internal::surfaceSetup(slave_edges,slave_coref_group,m_geometry);
  Internal::surfaceSetup(master_edges,master_coref_group,m_geometry);

  {
    Internal::AddEdges addEdges;
    for(std::map<IntPair, Integer>::iterator i=slave_edges.begin(); i!=slave_edges.end(); ++i)
      if (i->second == 1)
        addEdges(i->first);
      else if (i->second > 2)
        fatal() << "Not manifold slave surface";
    if (addEdges.connexCount() > 1)
      fatal() << "Slave surface is not connex";
  }
  {
    Internal::AddEdges addEdges;
    for(std::map<IntPair, Integer>::iterator i=master_edges.begin(); i!=master_edges.end(); ++i)
      if (i->second == 1)
        addEdges(i->first);
      else if (i->second > 2)
        fatal() << "Not manifold master surface";
    if (addEdges.connexCount() > 1)
      fatal() << "Master surface is not connex";
  }

  // 4- Calcul du co-raffinement par composantes connexes
  // Actuellement une seule composante par surface
  // A partir de cette étape si modif de maillage le résultat du coraffinement est invalidé
  // car une renumérotation des items locaux est possible 

  ISurfaceUtils * surfaceUtils = options()->surfaceUtils();
  ISurface * masterSurface = surfaceUtils->createSurface();
  surfaceUtils->setFaceToSurface(masterSurface,master_coref_group);
  ISurface * slaveSurface = surfaceUtils->createSurface();
  surfaceUtils->setFaceToSurface(slaveSurface,slave_coref_group);

  FaceFaceContactList coarse_contacts;
  surfaceUtils->computeSurfaceContact(masterSurface,slaveSurface,coarse_contacts);

//   { // test d'intégrité du co-raffinement local
//     Real3 normalA_coref(0,0,0), normalA_local(0,0,0);
//     Real3 normalB_coref(0,0,0), normalB_local(0,0,0);
    
//     ENUMERATE_FACE(iface,m_master_group.own())
//       normalA_local += m_geometry->computeOrientedMeasure(*iface);
//     ENUMERATE_FACE(iface,m_slave_group.own())
//       normalB_local += m_geometry->computeOrientedMeasure(*iface);
    
//     for(Integer i=0; i<coarse_contacts.size(); ++i) {
//       const ISurfaceUtils::FaceFaceContact & contact = coarse_contacts[i];
//       if (not contact.faceA.null() and contact.faceA.isOwn()) 
//         normalA_coref += contact.normalA;
//       if (not contact.faceB.null() and contact.faceB.isOwn()) 
//         normalB_coref += contact.normalB;
//     }    
//     pinfo() << "Global co-refinement stats0 : " 
//             << math::normeR3(normalA_coref) << " vs " <<  math::normeR3(normalA_local) << " ; "
//             << math::normeR3(normalB_coref) << " vs " <<  math::normeR3(normalB_local);
//   }

#ifndef NO_USER_WARNING
#warning "utiliser un destroySurface ???"
#endif /* NO_USER_WARNING */
  delete masterSurface;
  delete slaveSurface;

  // 5- Filtrage des contacts par la distance de leurs 2 faces
  // et diffusion des faces maitres utiles coté esclave.
  m_contacts.clear();

  // Enregistrement des besoins de raffinement pour des void-face
  ItemInternalList internals = mesh()->faceFamily()->itemsInternal();
  // La clef des SharedCorefinement est en référence à l'internals précédent
  typedef std::map<Integer,Real3> SharedCorefinement;
  SharedCorefinement voidface_shared_corefinement;
  ARCANE_ASSERT(( coarse_contacts.empty() == master_coref_group.empty() ),("Incompatible emptyness"));

  if (coarse_contacts.empty()) 
    {
      ENUMERATE_FACE(iface,m_slave_group.own()) {
        voidface_shared_corefinement[iface->localId()];
      }
    }
  else 
    {
      Internal::CheckCloseFaces checkFaces(nodesCoordinates,m_box_tolerance);
      SharedCorefinement facevoid_shared_corefinement; // pour aggregation sur le master
      for(FaceFaceContactList::iterator i = coarse_contacts.begin(); i != coarse_contacts.end(); ++i) {
        const ISurfaceUtils::FaceFaceContact & contact = *i;
        const Face master_face = contact.faceA;
        const Face slave_face = contact.faceB;
        if (slave_face.null()) {
          if (master_face.isOwn()) {
            facevoid_shared_corefinement[master_face.localId()] += contact.normalA;
          } // else not use since non local
        } else if (master_face.null()) {
          if (slave_face.isOwn()) {
            // Cette face doit etre post-traitée en parallèle
            voidface_shared_corefinement[slave_face.localId()];
          } // else not use since non local
        } else if (master_face.isOwn() or slave_face.isOwn()) {
          if (checkFaces(master_face,slave_face)) { 
            // Distance compatible avec la sélection de la boite
            // Contact admissible (stockage + préparation pour diffusion)
            if (master_face.isOwn()) // si !master_face.isOwn() il recevra l'info par son owner
              m_contacts.add(contact);
          } else {
            // Complétion du co-raffinement du non-contact
            if (master_face.isOwn()) {
              facevoid_shared_corefinement[master_face.localId()] += contact.normalA;
            }
            if (slave_face.isOwn()) {
              voidface_shared_corefinement[slave_face.localId()];
            }
          }
        }
      }

      // Intégration des face-void aggrégés
      for(SharedCorefinement::const_iterator i=facevoid_shared_corefinement.begin();
          i != facevoid_shared_corefinement.end(); ++i) {
        Face faceA(internals[i->first]);
        Face faceB; // null
        ARCANE_ASSERT((faceA.isOwn()),("Non local facevoid"));
        ISurfaceUtils::FaceFaceContact contact(faceA,faceB);
        contact.normalA = i->second;
        // info() << "Global co-refinement info : facevoid add " << faceA.uniqueId() << " " << contact.normalA;
        m_contacts.add(contact);
      }
    }

  // 6- Sérialisation des données vers les esclaves; préparation de la synthèse void-face

  // Compatibilité des données à envoyer
  // Comptabilise des sous-faces bi-parent (faceface)
  UniqueArray<Integer> info_to_send(nbSubDomain,0);
  for(FaceFaceContactList::iterator i = m_contacts.begin(); i != m_contacts.end(); ++i) 
    {
      const ISurfaceUtils::FaceFaceContact & contact = *i;
      const Face master_face = contact.faceA;
      const Face slave_face = contact.faceB;
      if (not slave_face.null()) {
        ARCANE_ASSERT((not master_face.null()),("Integrity error")); // pas construction du remplissage de m_contacts
        if (slave_face.isOwn()) {
          if (master_face.isOwn()) {
            // Traitement local des void-faces
            SharedCorefinement::iterator vf_finder = voidface_shared_corefinement.find(slave_face.localId());
            if (vf_finder != voidface_shared_corefinement.end()) {
              vf_finder->second += contact.normalB; 
              // info() << "Global co-ref Voidface add1 to slave " << slave_face.uniqueId() << " " << contact.normalB << " from " << master_face.uniqueId() << " " << contact.normalA;
            }
          }
        } else { // has to be sent
          if (master_face.isOwn())
            ++info_to_send[slave_face.owner()];
        }
      }
    }

  // Liste de synthèse des messages (emissions / réceptions)
  ISerializeMessageList * messageList = NULL;
  if (parallel_mng->isParallel())
    messageList = parallel_mng->createSerializeMessageList();

  // Préparation de la réception
  UniqueArray<Integer> info_to_recv(nbSubDomain,0);
  parallel_mng->allToAll(info_to_send,info_to_recv,1);
  std::list<std::shared_ptr<SerializeMessage> > recv_messages;
  for(Integer i=0;i<nbSubDomain;++i)
    {
      if (info_to_recv[i] > 0)
        {
          SerializeMessage * message = new SerializeMessage(parallel_mng->commRank(),i,ISerializeMessage::MT_Recv);
          recv_messages.push_back(std::shared_ptr<SerializeMessage>(message));
          messageList->addMessage(message);
        }
    }

  // Préparation des émissions
  UniqueArray<std::shared_ptr<SerializeMessage> > sent_messages(nbSubDomain);
  for(Integer i=0;i<nbSubDomain;++i)
    {
      const Integer faceface_info_to_send = info_to_send[i];
      if (faceface_info_to_send > 0)
        {
          SerializeMessage * message = new SerializeMessage(parallel_mng->commRank(),i,ISerializeMessage::MT_Send);
          sent_messages[i].reset(message);
          messageList->addMessage(message);
          SerializeBuffer & sbuf = message->buffer();
          sbuf.setMode(ISerializer::ModeReserve); // phase préparatoire
          sbuf.reserveInteger(2); // Nb d'item faceface puis voidface
          sbuf.reserve(DT_Int64,2*faceface_info_to_send); // Les uid
          sbuf.reserve(DT_Real,6*faceface_info_to_send); // Les normales (rien pour voidface)
          sbuf.reserve(DT_Real,6*faceface_info_to_send); // Les centres  (rien pour voidface)
          sbuf.allocateBuffer(); // allocation mémoire
          sbuf.setMode(ISerializer::ModePut);
          sbuf.putInteger(faceface_info_to_send);
        }
    }

  // Remplissage des messages sortants
  for(FaceFaceContactList::iterator i = m_contacts.begin(); i != m_contacts.end(); ++i)
    {
      const ISurfaceUtils::FaceFaceContact & contact = *i;
      const Face master_face = contact.faceA;
      const Face slave_face = contact.faceB;
      if (not slave_face.null() and not slave_face.isOwn() and master_face.isOwn())
        {
          std::shared_ptr<SerializeMessage> message = sent_messages[slave_face.owner()];
          SerializeBuffer & sbuf = message->buffer();
          // Pas possible de mettre des Real3 ??? (meme si on peut les réserver)
          sbuf.put(master_face.uniqueId().asInt64());
          sbuf.putReal(contact.normalA.x); sbuf.putReal(contact.normalA.y); sbuf.putReal(contact.normalA.z);
          sbuf.putReal(contact.centerA.x); sbuf.putReal(contact.centerA.y); sbuf.putReal(contact.centerA.z);
          sbuf.put(slave_face.uniqueId().asInt64());
          sbuf.putReal(contact.normalB.x); sbuf.putReal(contact.normalB.y); sbuf.putReal(contact.normalB.z);
          sbuf.putReal(contact.centerB.x); sbuf.putReal(contact.centerB.y); sbuf.putReal(contact.centerB.z);

          // pinfo() << "Global co-ref " << parallel_mng->commRank() << " push to " 
          // << slave_face.owner() << " " << master_face.uniqueId() << " - " << slave_face.uniqueId() 
          // << " : " << contact.normalA << " " << contact.normalB;
        }
    }

  // Traitement des communications
  if (messageList) {
    messageList->processPendingMessages();
    messageList->waitMessages(Parallel::WaitAll);
    delete messageList; messageList = NULL; // Destruction propre
  }

  // Traitement des messages reçus
  for(std::list<std::shared_ptr<SerializeMessage> >::iterator i=recv_messages.begin(); i!=recv_messages.end(); ++i)
    {
      std::shared_ptr<SerializeMessage> & message = *i;
      const Integer origDomainId = message->destRank();
      ARCANE_ASSERT((origDomainId != subDomain()->subDomainId()),("Local to local sent"));
      SerializeBuffer& sbuf = message->buffer();
      sbuf.setMode(ISerializer::ModeGet);
      const Integer faceface_count = sbuf.getInteger();
      ARCANE_ASSERT((faceface_count == info_to_recv[origDomainId]),("Bad recv faceface count"));
      Int64UniqueArray uids(2*faceface_count);
      Int32UniqueArray lids(2*faceface_count);
      sbuf.get(uids);
      mesh()->faceFamily()->itemsUniqueIdToLocalId(lids,uids,true);

      // Traitement des pairs face-face
      for(Integer j=0;j<faceface_count;++j)
        {
          Face faceA(internals[lids[2*j  ]]);
          Face faceB(internals[lids[2*j+1]]);
          ARCANE_ASSERT((faceB.isOwn()),("Not local information recieved"));
          ISurfaceUtils::FaceFaceContact contact(faceA,faceB);
          contact.normalA.x = sbuf.getReal(); contact.normalA.y = sbuf.getReal(); contact.normalA.z = sbuf.getReal();
          contact.centerA.x = sbuf.getReal(); contact.centerA.y = sbuf.getReal(); contact.centerA.z = sbuf.getReal();
          contact.normalB.x = sbuf.getReal(); contact.normalB.y = sbuf.getReal(); contact.normalB.z = sbuf.getReal();
          contact.centerB.x = sbuf.getReal(); contact.centerB.y = sbuf.getReal(); contact.centerB.z = sbuf.getReal();
          m_contacts.add(contact);

          SharedCorefinement::iterator vf_finder = voidface_shared_corefinement.find(faceB.localId());
          if (vf_finder != voidface_shared_corefinement.end()) {
            vf_finder->second += contact.normalB; 
            // info() << "Global co-ref Voidface add2 to slave " << faceB.uniqueId() << " " << contact.normalB << " from " << faceA.uniqueId() << " " << contact.normalA;
          }
        }
    }

  // 7- Traitement final des pairs void-face
  for(SharedCorefinement::const_iterator i=voidface_shared_corefinement.begin();
      i != voidface_shared_corefinement.end(); ++i) {
    Face faceA; // null
    Face faceB(internals[i->first]);
    ARCANE_ASSERT((faceB.isOwn()),("Non local voidface"));
    Real3 faceB_normal = m_geometry->computeOrientedMeasure(faceB);
    Real faceB_area = math::normeR3(faceB_normal);
    if (math::normeR3(i->second) < faceB_area) {
      ISurfaceUtils::FaceFaceContact contact(faceA,faceB);
      contact.normalB = faceB_normal - i->second;
      // info() << "Global co-refinement info : voidface add " << faceB.uniqueId() << " " << contact.normalB << "    " << faceB_normal;
      m_contacts.add(contact);
    }
  }

  // 8- Nettoyage des réplicats inutiles
  // Non implémentés et ATTN cela peut modifier les numérotations locales

  m_internal->corefinementTimer.stop();
 
  UniqueArray<Real> times(2);
  times[0] = m_internal->corefinementTimer.totalTime();
  times[1] = m_internal->migrationTimer.totalTime();
  parallel_mng->reduce(Parallel::ReduceMin,times);
  info() << "Corefinement timers : all=" << times[0] << " migration=" << times[1] << " #domains=" << parallel_mng->commSize();
}

/*---------------------------------------------------------------------------*/

const IParallelCorefinement::FaceFaceContactList & 
ParallelCorefinementService::
contacts()
{
  return m_contacts;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_PARALLELCOREFINEMENT(ParallelCorefinement,ParallelCorefinementService);
