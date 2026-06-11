// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SplitSDMeshPartitioner.cc                                   (C) 2000-2024 */
/*                                                                           */
/* Mesh partitioner reproducing the (simplified) functionality of SplitSD    */
/* used at Dassault Aviation and developed at ONERA by EB in 1996-99         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//#define DEBUG_PARTITIONER

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ArcanePrecomp.h"
#include "arcane/utils/Convert.h"
#include "arcane/utils/Array.h"

#include "arcane/core/ISubDomain.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/ItemEnumerator.h"
#include "arcane/core/IPrimaryMesh.h"
#include "arcane/core/Properties.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/Service.h"
#include "arcane/core/Timer.h"
#include "arcane/core/FactoryService.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/MeshVariable.h"
#include "arcane/core/VariableBuildInfo.h"
#include "arcane/std/MeshPartitionerBase.h"
#include "arcane/std/SplitSDMeshPartitioner.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
SplitSDMeshPartitioner::
SplitSDMeshPartitioner(const ServiceBuildInfo& sbi)
: ArcaneSplitSDMeshPartitionerObject(sbi)
, m_poids_aux_mailles(VariableBuildInfo(sbi.mesh(), "MeshPartitionerCellsWeight", IVariable::PNoDump | IVariable::PNoRestore))
{
  info() << "SplitSDMeshPartitioner::SplitSDMeshPartitioner(...)";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SplitSDMeshPartitioner::
partitionMesh(bool initial_partition)
{
  info() << "Load balancing with SplitSDMeshPartitioner";

  // we do not support constraints
  // because we use the Arcane mesh for frontal sweeps,
  // it is therefore not planned for the moment to take into account groups
  // of cells associated with constraints
  _initArrayCellsWithConstraints();
  if (haveConstraints())
    throw FatalErrorException("SplitSDMeshPartitioner: Constraints are not supported with SplitSD");

  // initialization of internal structures

  StrucInfoProc* InfoProc = NULL; /* structure describing the processor on which the application runs */
  StructureBlocEtendu* Domaine = NULL; /* structure summarizing the block, i.e., the part of the topology localized on this processor */
  StrucMaillage* Maillage = NULL; /* structure summarizing the global mesh */

  // initialization of partitioner-specific data, m_cells_weight must be calculated
  init(initial_partition, InfoProc, Domaine, Maillage);

  // iterative load balancing process by moving elements from one subdomain to another
  Equilibrage(InfoProc, Domaine, Maillage);

  // we clear the structures
  fin(InfoProc, Domaine, Maillage);
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
void SplitSDMeshPartitioner::
init(bool initial_partition, StrucInfoProc*& InfoProc, StructureBlocEtendu*& Domaine, StrucMaillage*& Maillage)
{
  info() << "SplitSDMeshPartitioner::init(" << initial_partition << ",...)";

  ISubDomain* sd = subDomain();
  IParallelMng* pm = sd->parallelMng();

  /* Memory initialization of InfoProc  */
  InfoProc = new StrucInfoProc();

  InfoProc->me = sd->subDomainId();
  InfoProc->nbSubDomain = pm->commSize();
  InfoProc->m_service = this;
  // MPI_Comm_dup(MPI_COMM_WORLD,&InfoProc->Split_Comm); /* Assignment of a communicator specific to our partitioner */
  InfoProc->Split_Comm = *(MPI_Comm*)getCommunicator();
  initConstraints();

  // initialization of cell weights
  initPoids(initial_partition);
#if 0
  if (!initial_partition){
    info() << "Initialize new owners";
    // Initialise the new owner for the case where no rebalancing is needed
    IMesh* mesh = this->mesh();
    VariableItemInteger& cells_new_owner = mesh->itemsNewOwner(IK_Cell);
    ENUMERATE_CELL(icell,mesh->ownCells()){
      Cell cell = *icell;
      cells_new_owner[icell] = cell.owner();
    }
    changeOwnersFromCells();
    cells_new_owner.synchronize();
  }
#endif

  /* Memory initialization of Domain */
  Domaine = new StructureBlocEtendu();

  Domaine->NbIntf = 0;
  Domaine->Intf = NULL;
  Domaine->NbElements = 0;
  Domaine->PoidsDom = 0.0;

  MAJDomaine(Domaine);

  /* Memory initialization of Mesh */
  Maillage = new StrucMaillage();

  Maillage->NbElements = 0;
  Maillage->NbDomainesMax = pm->commSize();
  Maillage->NbDomainesPleins = 0;
  Maillage->ListeDomaines = NULL;
  Maillage->NbProcsVides = 0;
  Maillage->Poids = 0.0;

  // we only update the structure on the master processor (0), based on Domain
  MAJMaillageMaitre(InfoProc, Domaine, Maillage);
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
void SplitSDMeshPartitioner::
initPoids(bool initial_partition)
{
  ARCANE_UNUSED(initial_partition);

  /* retrieval of cell weights so that they follow the cells */
  IMesh* mesh = this->mesh();
  CellGroup own_cells = mesh->ownCells();
  //   if (initial_partition || m_cells_weight.empty())
  //     ENUMERATE_CELL(iitem,own_cells){
  //       const Cell& cell = *iitem
  //       m_poids_aux_mailles[cell] = 1.0;
  //     }
  //   else{
  //     Integer nb_weight = nbCellWeight();
  SharedArray<float> cell_weights = cellsWeightsWithConstraints(1, true); // 1 weight only
  ENUMERATE_CELL (iitem, own_cells) {
    m_poids_aux_mailles[iitem] = cell_weights[iitem.index()];
  }
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
void SplitSDMeshPartitioner::
fin(StrucInfoProc*& InfoProc, StructureBlocEtendu*& Domaine, StrucMaillage*& Maillage)
{
  info() << "SplitSDMeshPartitioner::fin(...)";
  LibereInfoProc(InfoProc);
  LibereDomaine(Domaine);
  LibereMaillage(Maillage);

  delete InfoProc;
  delete Domaine;
  delete Maillage;

  InfoProc = NULL;
  Domaine = NULL;
  Maillage = NULL;
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
void SplitSDMeshPartitioner::
MAJDomaine(StructureBlocEtendu* Domaine)
{
  debug() << " ----------------------------------------";
  debug() << "SplitSDMeshPartitioner::MAJDomaine(...)";

  LibereDomaine(Domaine);

  int me = subDomain()->subDomainId();

  // filling Domaine->Intf, lists of nodes per neighboring subdomain

  IMesh* mesh = this->mesh();
  FaceGroup all_faces = mesh->allFaces(); // faces on this processor
  debug() << " all_faces.size() = =" << all_faces.size();

  std::map<int, SharedArray<Face>> vois_faces; // list of faces per neighboring subdomain

  ENUMERATE_FACE (i_item, all_faces) {
    const Face face = *i_item;
    if (!face.isSubDomainBoundary()) {
      int id1 = face.backCell().owner();
      int id2 = face.frontCell().owner();

      if (id1 != id2) { // case of neighborhood with another domain
        int idv = -1; // neighbor number
        if (id1 == me)
          idv = id2;
        else if (id2 == me)
          idv = id1;
        else
          continue;
        // info() << "idv = "<<idv<< " for the face " <<face.uniqueId();

        // we add the face index for a subdomain
        SharedArray<Face>& v_face = vois_faces[idv];
        v_face.add(face);

      } // end if (id1 != id2)
    } // end if (!face.isBoundary())
  } // end ENUMERATE_FACE

  UniqueArray<int> filtreNoeuds(mesh->nodeFamily()->maxLocalId());
  filtreNoeuds.fill(0); // initialization of the filter
  int marque = 0;

  // for each neighboring subdomain, we search for nodes in the interface
  // avoiding duplicates using the node filter

  Domaine->NbIntf = arcaneCheckArraySize(vois_faces.size());

  // cells on this proc (without ghost cells)
  Domaine->NbElements = mesh->ownCells().size();

  // calculation of the total weight of a domain => using m_cells_weight
  Domaine->PoidsDom = 0.0;
  CellGroup own_cells = mesh->ownCells();
  ENUMERATE_CELL (iitem, own_cells) {
    Cell cell = *iitem;
    Domaine->PoidsDom += m_poids_aux_mailles[cell];
  }

#ifdef ARCANE_DEBUG
  info() << "Domaine->NbIntf = " << Domaine->NbIntf;
  info() << "Domaine->NbElements = " << Domaine->NbElements;
  info() << "Domaine->PoidsDom = " << Domaine->PoidsDom;
#endif

  Domaine->Intf = new StructureInterface[Domaine->NbIntf];
  unsigned int ind = 0;
  for (std::map<int, SharedArray<Face>>::iterator iter_vois = vois_faces.begin();
       iter_vois != vois_faces.end(); ++iter_vois) {

    marque += 1;

    Domaine->Intf[ind].NoDomVois = (*iter_vois).first;
    Array<Face>& v_face = (*iter_vois).second;

#ifdef ARCANE_DEBUG
    info() << "Domaine->Intf[" << ind << "].NoDomVois = " << Domaine->Intf[ind].NoDomVois;
    info() << " v_face.size() = " << v_face.size();
#endif

    for (int i = 0; i < v_face.size(); i++) {
      const Face& face = v_face[i];
      //       info() << " v_face["<<i<<"].uniqueId() = " << face.uniqueId()
      // 	     << ", backCell: "<<face.backCell().owner() <<", frontCell: "<< face.frontCell().owner();

      for (Integer z = 0; z < face.nbNode(); ++z) {
        const Node node = face.node(z);
        Integer node_local_id = node.localId();
        if (filtreNoeuds[node_local_id] != marque) {
          Domaine->Intf[ind].ListeNoeuds.add(node);
          filtreNoeuds[node_local_id] = marque;
        }
      }

    } // end for iter_face
#ifdef ARCANE_DEBUG
    info() << " ListeNoeuds.size() = " << Domaine->Intf[ind].ListeNoeuds.size();
#endif

    ind += 1;
  } // end for iter_vois

#ifdef ARCANE_DEBUG
  info() << "MAJDomaine => ";
  AfficheDomaine(1, Domaine);
  info() << " ----------------------------------------";
#endif
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
void SplitSDMeshPartitioner::
MAJMaillageMaitre(StrucInfoProc* InfoProc, StructureBlocEtendu* Domaine, StrucMaillage* Maillage)
{
  // prerequisite: having done a MAJDomaine before coming here

#ifdef ARCANE_DEBUG
  info() << "SplitSDMeshPartitioner::MAJMaillageMaitre(...)";
  LibereMaillage(Maillage);
#endif

  void* TabTMP; /* array for Domain => Mesh->ListeDomaines[*] communications */
  int TailleTab; /* size of the TabTMP array (in bytes) */
  int iDom; /* loop index over domains */

  /* we concentrate the information in TabTMP */
  TailleTab = TailleDom(Domaine);
  TabTMP = malloc((size_t)TailleTab);

  PackDom(InfoProc, Domaine, TabTMP, TailleTab, InfoProc->Split_Comm);

  /* we send to the master, for all domains
     (except the master which just copies) */
  if (InfoProc->me == 0) {
#ifdef DEBUG
    info() << "   ***************************";
    info() << "   * Before MAJMaillageMaitre *";
    AfficheMaillage(Maillage);
    info() << "   ***************************";
#endif

    if (Maillage->ListeDomaines == NULL) {
      Maillage->ListeDomaines = new StrucListeDomMail[InfoProc->nbSubDomain];
      for (iDom = 0; iDom < InfoProc->nbSubDomain; iDom++) {
        Maillage->ListeDomaines[iDom].NbElements = 0;
        Maillage->ListeDomaines[iDom].Poids = 0;
        Maillage->ListeDomaines[iDom].NbVoisins = 0;
        Maillage->ListeDomaines[iDom].ListeVoisins = NULL;
      }
    } /* end if Maillage->ListeDomaines == NULL */

    UnpackDom(TabTMP, TailleTab, InfoProc->Split_Comm, &Maillage->ListeDomaines[0]);
  }
  else { /* case where it is not the master */
    EnvoieMessage(InfoProc, 0, TAG_MAILLAGEMAITRE, TabTMP, TailleTab);
  }

  /* release */
  free(TabTMP);
  TabTMP = NULL;

  /* --------- */
  /* Reception */
  /* --------- */

  /* the master receives all information from other nodes (including itself) */
  if (InfoProc->me == 0) {
    Maillage->NbDomainesPleins = 0; /* we will count the number of full domains */
    Maillage->NbProcsVides = 0; /* we will count the number of empty domains */
    Maillage->Poids = 0.0; // total weight
    Maillage->NbElements = 0; // total number of elements in the mesh

    /* we put the smallest numbers at the end to use them first */
    for (iDom = InfoProc->nbSubDomain - 1; iDom >= 0; iDom--) {
      if (iDom != 0) {

        TailleTab = 0; /* because we do not yet know the size */
        TabTMP = RecoitMessage(InfoProc, iDom, TAG_MAILLAGEMAITRE, &TailleTab);

        UnpackDom(TabTMP, TailleTab, InfoProc->Split_Comm, &Maillage->ListeDomaines[iDom]);

        free((void*)TabTMP);
        TabTMP = NULL;
      }

      /* we count the number of full and empty domains */
      if (Maillage->ListeDomaines[iDom].NbElements != 0)
        Maillage->NbDomainesPleins += 1;
      else
        Maillage->NbProcsVides += 1;

      Maillage->Poids += Maillage->ListeDomaines[iDom].Poids;
      Maillage->NbElements += Maillage->ListeDomaines[iDom].NbElements;
    }

#ifdef ARCANE_DEBUG
    info() << "   ***************************";
    info() << "   * After MAJMaillageMaitre *";
#ifdef DEBUG
    AfficheMaillage(Maillage);
#endif
    info() << "   ***************************";
    verifMaillageMaitre(Maillage);
#endif
  } /* end if me == master */
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
void SplitSDMeshPartitioner::
verifMaillageMaitre(StrucMaillage* Maillage)
{
#ifdef ARCANE_DEBUG
  info() << "  entering verifMaillageMaitre";

  StrucListeDomMail* ListeDomaines = Maillage->ListeDomaines;
  int NbDomaines = Maillage->NbDomainesPleins;

  for (int iDom = 0; iDom < NbDomaines; iDom++) {
    for (int j = 0; j < ListeDomaines[iDom].NbVoisins; j++) {

      int NbNoeudsInterface = ListeDomaines[iDom].ListeVoisins[j].NbNoeudsInterface;
      int iVois = ListeDomaines[iDom].ListeVoisins[j].NoDomVois;

      // search if the neighbor exists and if the number of nodes is identical
      int k;
      for (k = 0; k < ListeDomaines[iVois].NbVoisins && ListeDomaines[iVois].ListeVoisins[k].NoDomVois != iDom; k++) {
      }
      if (k == ListeDomaines[iVois].NbVoisins) {
        printf("we cannot find the neighbor number \n");
        printf("for info: iDom = %d, iVois  = %d\n", iDom, iVois);
        perror() << "verifMaillageMaitre error on neighborhood !!!";
      }

      if (ListeDomaines[iVois].ListeVoisins[k].NbNoeudsInterface != NbNoeudsInterface) {
        printf("we do not find the same number of nodes between neighbors \n");
        printf("for info: iDom = %d, iVois  = %d, NbNoeudsInterface %d != %d\n", iDom, iVois, NbNoeudsInterface, ListeDomaines[iVois].ListeVoisins[k].NbNoeudsInterface);
        perror() << "verifMaillageMaitre error on number of nodes !!!";
      }
    }
  }

  info() << "  leaving verifMaillageMaitre";
#else
  ARCANE_UNUSED(Maillage);
#endif
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
void SplitSDMeshPartitioner::
MAJDeltaGlobal(StrucInfoProc* InfoProc, StrucMaillage* Maillage, double tolerance)
{
  StrucListeDomMail* ListeDomaines = Maillage->ListeDomaines;
  int NbDomaines = Maillage->NbDomainesPleins;

  int i;
  int iDomDep; /* index of the full domain that serves as the starting point for front sweeps */
  double PoidsMoyen;

  int tailleFP;
  int tailleFS;

  int* FrontPrec; // front of nodes of the graph (domains)
  int* FrontSuiv;
  int* FrontTMP;
  /* for each node in the graph, gives the previous node.
     This will allow, from a given node during the front sweep, to find a path back to the starting node */
  int* ListeNoeudsPrec;

  int* FiltreDomaine; /* to mark the domains as markedDomVu, markedDomNonVu, or markedDomVide */
  int marqueDomVu = 1;
  int marqueDomNonVu = 0;
  int marqueDomVide = -1; // normally, there are none

  double* PoidsSave;

#ifdef ARCANE_DEBUG
  info() << " ------------------------------------";
  info() << " entering MAJDeltaGlobal, tolerance = " << tolerance;
  info() << " ------------------------------------";
  info() << " ......... Initial Mesh ...........";
  AfficheListeDomaines(ListeDomaines, NbDomaines);
#endif

  if (NbDomaines == 0) {
    info() << " SplitSDMeshPartitioner::MAJDeltaGlobal: NbDomaines is null!";
    return;
  }
  if (Maillage->NbDomainesPleins <= 1) {
#ifdef ARCANE_DEBUG
    info() << " \n = exiting MAJDeltaGlobal without doing anything (NbDomainesPleins = " << Maillage->NbDomainesPleins;
    info() << " ------------------------------------";
#endif
    return;
  }

  FiltreDomaine = (int*)malloc((size_t)NbDomaines * sizeof(int));
  CHECK_IF_NOT_NULL(FiltreDomaine);

  /* setting all Deltas to zero for safety */
  for (i = 0; i < NbDomaines; i++)
    for (int j = 0; j < ListeDomaines[i].NbVoisins; j++)
      ListeDomaines[i].ListeVoisins[j].Delta = 0.0;

  /*
     shifting the number of nodes of each domain so that the total sum is <= 0
  */

  for (i = 0; i < NbDomaines; i++) {
    if (ListeDomaines[i].NbElements != 0)
      FiltreDomaine[i] = marqueDomNonVu;
    else
      FiltreDomaine[i] = marqueDomVide;
  }

  PoidsSave = (double*)malloc((size_t)NbDomaines * sizeof(double));
  CHECK_IF_NOT_NULL(PoidsSave);

  /* the average value */
  PoidsMoyen = Maillage->Poids / (double)Maillage->NbDomainesMax;

  /* performing the shift */
  for (i = 0; i < NbDomaines; i++) {
    /* saving the weight */
    PoidsSave[i] = ListeDomaines[i].Poids;

    if (FiltreDomaine[i] != marqueDomVide)
      ListeDomaines[i].Poids -= PoidsMoyen;
  }

#ifdef DEBUG
  info() << " After Poids -= PoidsMoyen";
  AfficheListeDomaines(ListeDomaines, NbDomaines);
#endif

  /* memory initialization to the largest size */
  FrontPrec = (int*)malloc((size_t)(NbDomaines - 1) * sizeof(int));
  CHECK_IF_NOT_NULL(FrontPrec);
  FrontSuiv = (int*)malloc((size_t)(NbDomaines - 1) * sizeof(int));
  CHECK_IF_NOT_NULL(FrontSuiv);

  ListeNoeudsPrec = (int*)malloc((size_t)(NbDomaines) * sizeof(int));
  CHECK_IF_NOT_NULL(ListeNoeudsPrec);

  for (iDomDep = 0; iDomDep < NbDomaines; iDomDep++) {
#ifdef DEBUG
    info() << " ListeDomaines[iDomDep = " << iDomDep << "].Poids  = " << ListeDomaines[iDomDep].Poids;
#endif

    /* we only take domains above the average weight (so > 0 now) as starting points */
    if (ListeDomaines[iDomDep].Poids > tolerance) {
      /* BELOW, A NODE IS A DOMAIN; IT IS A NODE OF THE GRAPH */

      /* initializing previously seen domains to non-seen */
      for (i = 0; i < NbDomaines; i++)
        if (FiltreDomaine[i] == marqueDomVu)
          FiltreDomaine[i] = marqueDomNonVu;

      /* marking the current node as seen */
      FiltreDomaine[iDomDep] = marqueDomVu;

      /* initialization of the starting front */
      tailleFS = 1;
      FrontSuiv[tailleFS - 1] = iDomDep;

#ifdef DEBUG
      info() << " FrontSuiv[0] = " << iDomDep;
#endif

      /* initialization of the starting node (no previous node) */
      ListeNoeudsPrec[FrontSuiv[tailleFS - 1]] = -1;

      /* loop while the starting node is too large (weight>0) */
      while (ListeDomaines[iDomDep].Poids > tolerance) {
#ifdef DEBUG
        info() << " while (ListeDomaines[" << iDomDep << "].Poids  = " << ListeDomaines[iDomDep].Poids << " > " << tolerance;
#endif

        /* swapping the following and preceding fronts */
        FrontTMP = FrontPrec;
        FrontPrec = FrontSuiv;
        FrontSuiv = FrontTMP;
        tailleFP = tailleFS;
        tailleFS = 0; /* the following front is now empty */

        /* case where the starting domain failed to disperse its surplus onto other domains, and we have seen everything we could */
        if (tailleFP == 0) {
          fatal() << " partitionner/MAJDeltaGlobal: no more domains found while not finished!!!";
        }

        /*
	   progressing one front
	*/

        /* loop over the nodes of the preceding front */
        for (int iFP = 0; iFP < tailleFP; iFP++) {
          int iDom = FrontPrec[iFP];

          /* loop over the neighbors of the node */
          for (int iVois = 0; iVois < ListeDomaines[iDom].NbVoisins; iVois++) {
            int iDomVois = ListeDomaines[iDom].ListeVoisins[iVois].NoDomVois;
            if (FiltreDomaine[iDomVois] == marqueDomNonVu) {
              /* marking this node */
              FiltreDomaine[iDomVois] = marqueDomVu;
              ListeNoeudsPrec[iDomVois] = iDom;
#ifdef DEBUG
              info() << "  FrontSuiv[" << tailleFS << "] = " << iDomVois;
#endif
              FrontSuiv[tailleFS++] = iDomVois;
            }
          }
        } /* end for iFP<tailleFP */

        /*
	   discharging the base node onto the nodes of the following front
	*/

        /* loop over the nodes of the following front */
        for (int iFS = 0; iFS < tailleFS; iFS++) {
          int iDom = FrontSuiv[iFS];

#ifdef DEBUG
          info() << " ListeDomaines[" << iDom << "].Poids = " << ListeDomaines[iDom].Poids;
#endif

          /* only giving to deficit nodes */
          if (ListeDomaines[iDom].Poids < 0.0) {
            /* not giving more than we have */
            double don = MIN(-ListeDomaines[iDom].Poids, ListeDomaines[iDomDep].Poids);

            /* case where there is a donation */
            if (don > 0.0) {
              int iDomTmp;
              int iDomTmpPrec;

              ListeDomaines[iDom].Poids += don;
              ListeDomaines[iDomDep].Poids -= don;

              //fprintf(stdout," donation = %f\n",don);

              /* tracing back the entire path to update the Delta
		 between nodes iDomTmp and iDomTmpPrec */
              iDomTmp = iDom;
              iDomTmpPrec = ListeNoeudsPrec[iDomTmp];
              while (iDomTmpPrec != -1) {
                /* incrementing the Delta by donation between iDomTmpPrec and iDomTmp */
                MAJDelta(don, iDomTmpPrec, iDomTmp, ListeDomaines);
                /* similarly (with the opposite sign) for the reciprocal interface */
                MAJDelta(-don, iDomTmp, iDomTmpPrec, ListeDomaines);

                /* moving one step back along the path */
                iDomTmp = iDomTmpPrec;
                iDomTmpPrec = ListeNoeudsPrec[iDomTmp];

              } /* end while iDomTmpPrec != -1 */

            } /* end if don != 0 */
          } /* end if NbNoeuds < 0, for the front domain */
        } /* end for iFS<tailleFS */
        //	AfficheListeDomaines(ListeDomaines,NbDomaines);

      } /* end while Poids > tolerance */
    } /* end if Poids > tolerance */
  } /* end for iDomDep<NbDomaines */

  /* restoring the values */
  for (i = 0; i < NbDomaines; i++)
    ListeDomaines[i].Poids = PoidsSave[i];

  free((void*)FrontPrec);
  FrontPrec = NULL;
  free((void*)FrontSuiv);
  FrontSuiv = NULL;
  free((void*)ListeNoeudsPrec);
  ListeNoeudsPrec = NULL;
  free((void*)FiltreDomaine);
  FiltreDomaine = NULL;
  free((void*)PoidsSave);
  PoidsSave = NULL;

#ifdef ARCANE_DEBUG
  info() << " ......... Final Mesh ...........";
  AfficheListeDomaines(ListeDomaines, NbDomaines);
  info() << " = exiting MAJDeltaGlobal =";
  info() << " ------------------------------------";
#endif
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
void SplitSDMeshPartitioner::
MAJDelta(double ajout, int iDom, int iVois, StrucListeDomMail* ListeDomaines)
{
  int j;
  for (j = 0;
       j < ListeDomaines[iDom].NbVoisins &&
       ListeDomaines[iDom].ListeVoisins[j].NoDomVois != iVois;
       j++) {
  }

  if (j == ListeDomaines[iDom].NbVoisins) {
    info() << "could not find the neighbor number";
    info() << "for info: addition = " << ajout << ", iDom = " << iDom << ", iVois = " << iVois;
    pfatal() << "Error in Partitionner/MAJDelta, no neighbor found!";
  }

  /* performing the shift */
  ListeDomaines[iDom].ListeVoisins[j].Delta += ajout;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
double SplitSDMeshPartitioner::
CalculDeltaMin(StrucMaillage* Maillage, double deltaMin, int iterEquilibrage, int NbMaxIterEquil)
{
  // the goal: limit deltaMin so that it is not too small the first time transfers are important.
  // Indeed, this tends to fragment the domains, which results in poorer performance (more waiting during communications)

  debug() << "  entering SplitSDMeshPartitioner::CalculDeltaMin,  deltaMin = " << deltaMin
          << ", iterEquilibrage = " << iterEquilibrage
          << ", NbMaxIterEquil = " << NbMaxIterEquil;
  if (arcaneIsDebug())
    AfficheMaillage(Maillage);

  double deltaAjuste = deltaMin;

  // searching for the largest Delta on the interfaces
  double deltaMaxItf = 0.0;
  // same for the sum of deltas per domain, normalized by the domain weight (so a ratio)
  double ratioDeltaMax = 0.0;

  StrucListeDomMail* ListeDomaines = Maillage->ListeDomaines;
  int NbDomaines = Maillage->NbDomainesPleins;

  for (int iDom = 0; iDom < NbDomaines; iDom++) {
    double deltaTotalDom = 0.0;
    for (int j = 0; j < ListeDomaines[iDom].NbVoisins; j++) {
      double delta = ListeDomaines[iDom].ListeVoisins[j].Delta;
      deltaMaxItf = MAX(delta, deltaMaxItf);
      if (delta > 0.0)
        deltaTotalDom += delta;
    }
    double ratio = 0.0;
    if (ListeDomaines[iDom].Poids > 0.0)
      ratio = deltaTotalDom / ListeDomaines[iDom].Poids;
    ratioDeltaMax = MAX(ratio, ratioDeltaMax);
  }

  double poidsMoy = Maillage->Poids / (double)Maillage->NbDomainesMax;

  // heuristic to limit deltaAjuste
  if (ratioDeltaMax > 0.9)
    deltaAjuste = poidsMoy / 3.0;
  else if (ratioDeltaMax > 0.5)
    deltaAjuste = deltaMaxItf / 10.0;

  // to avoid being smaller than the minimum which seems reasonable and is parameterized by the .plt file
  deltaAjuste = MAX(deltaMin, deltaAjuste);

#if defined(ARCANE_DEBUG) || defined(DEBUG_PARTITIONER)
  // what does this delta / average weight represent? all or a small proportion?
  double proportion = deltaMaxItf / poidsMoy;

  info() << " deltaMaxItf = " << deltaMaxItf;
  info() << " ratioDeltaMax = " << ratioDeltaMax;
  info() << " poidsMoy = " << poidsMoy;
  info() << " proportion = " << proportion;
  info() << " deltaAjuste = " << deltaAjuste;
#endif

  return deltaAjuste;
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
void SplitSDMeshPartitioner::
Equilibrage(StrucInfoProc* InfoProc, StructureBlocEtendu* Domaine, StrucMaillage* Maillage)
{
  int iDom; // index for loops
  int indDomCharge = -1; // the domain from which elements will be extracted
  int indDomVois = -1; // the domain that will receive from indDomCharge

  double deltaMin; // we will only perform the transfer for a minimum weight.
  // maximum imbalance (0.01) in .plt file => maxImbalance()
  deltaMin = Maillage->Poids / (double)Maillage->NbDomainesMax * maxImbalance() / 6.0; // we take maxImbalance/6 of the average as the value (so 0.1 by default)

  double poidsMax = 0; // for searching the most loaded domain

  void* TabTMP; /* to transfer info about the chosen procs and the master's Delta to the other procs */

  int NbAppelsAEquil2Dom = -1;

  int iterEquilibrage = 0;
  int NbMaxIterEquil = 5; // global balancing phase. TODO see if this value needs to be changed (parameterize it)

  double tolConnexite = 0.1; // a non-connected subdomain smaller than tolConnexite*average_size will be transferred

#ifdef ARCANE_DEBUG
  info() << " -------------------------------------";
  info() << " entering SplitSDMeshPartitioner::Equilibrage,  deltaMin = " << deltaMin;
#endif

  int TailleTMP = TailleEquil();
  TabTMP = malloc((size_t)TailleTMP);
  CHECK_IF_NOT_NULL(TabTMP);

  /* limiting to NbMaxIter iterations */
  while (iterEquilibrage < NbMaxIterEquil && NbAppelsAEquil2Dom != 0) {
    int* FiltreDomaine;
    int marqueDomVu = 1;
    int marqueDomNonVu = 0;

    iterEquilibrage += 1;
    NbAppelsAEquil2Dom = 0;
    poidsMax = 0;
    indDomCharge = -1;

    int* MasqueDesNoeuds = GetMasqueDesNoeuds(InfoProc);
    int* MasqueDesElements = GetMasqueDesElements(InfoProc);
    int marqueVu = 0;
    int marqueNonVu = 0;

    IPrimaryMesh* mesh = this->mesh()->toPrimaryMesh();

    /* on the master */
    if (InfoProc->me == 0) {
      info() << " SplitSDMeshPartitioner::Equilibrage of the load (iteration No " << iterEquilibrage << ")";

      /* updating Deltas on the domain description interfaces on the master */
      MAJDeltaGlobal(InfoProc, Maillage, deltaMin); // e.g. GetDeltaNoeuds

      // calculating a deltaMin based on local transfers
      double deltaAjuste = CalculDeltaMin(Maillage, deltaMin, iterEquilibrage, NbMaxIterEquil);

      /* marking the seen (or empty) domains */
      FiltreDomaine = (int*)calloc((size_t)Maillage->NbDomainesMax, sizeof(int));
      CHECK_IF_NOT_NULL(FiltreDomaine);

      // for empty domains
      for (iDom = 0; iDom < Maillage->NbDomainesMax; iDom++)
        if (Maillage->ListeDomaines[iDom].NbElements == 0)
          FiltreDomaine[iDom] = marqueDomVu;

      // loop while we find a domain to transfer from
      do {

        /* searching for the most loaded of the remaining domains */
        indDomCharge = -1;
        poidsMax = 0;

        for (iDom = 0; iDom < Maillage->NbDomainesMax; iDom++)
          if (FiltreDomaine[iDom] == marqueDomNonVu && Maillage->ListeDomaines[iDom].Poids > poidsMax) {
            poidsMax = Maillage->ListeDomaines[iDom].Poids;
            indDomCharge = iDom;
          }

#ifdef ARCANE_DEBUG
        info() << "indDomCharge = " << indDomCharge << "; poidsMax = " << poidsMax;
#endif

        if (indDomCharge != -1) {
          /* we no longer want to find it */
          FiltreDomaine[indDomCharge] = marqueDomVu;

          /* choosing 2 domains to perform the interface movement between the 2 */

          /* for this loaded domain, we look at which other domain there might be a node transfer */
          for (int i = 0; i < Maillage->ListeDomaines[indDomCharge].NbVoisins; i++) {
            indDomVois = Maillage->ListeDomaines[indDomCharge].ListeVoisins[i].NoDomVois;

            /* we limit ourselves to transfers where DeltaNoeuds > deltaNoeudsMin */
            if (Maillage->ListeDomaines[indDomCharge].ListeVoisins[i].Delta > deltaAjuste) {
#if defined(ARCANE_DEBUG) || defined(DEBUG_PARTITIONER)
              info() << " Balancing (" << iterEquilibrage << ") for the pair indDomCharge = " << indDomCharge << "; indDomVois = " << indDomVois
                     << "; Delta = " << Maillage->ListeDomaines[indDomCharge].ListeVoisins[i].Delta;
#endif

              /* We diffuse the information to all processors */
              PackEquil(InfoProc, indDomCharge, indDomVois, Maillage->ListeDomaines[indDomCharge].ListeVoisins[i].Delta,
                        TabTMP, TailleTMP, InfoProc->Split_Comm);

              TabTMP = DiffuseMessage(InfoProc, 0, TabTMP, TailleTMP);

              // Change the mark for each call (which allows the nodes to be released)
              marqueVu += 1;

              // Balancing between 2 domains
              Equil2Dom(MasqueDesNoeuds, MasqueDesElements, marqueVu, marqueNonVu,
                        InfoProc, Domaine, Maillage, indDomCharge, indDomVois,
                        Maillage->ListeDomaines[indDomCharge].ListeVoisins[i].Delta);
              NbAppelsAEquil2Dom += 1;

            } /* end if Delta > deltaAjuste */
          } /* end for i<NbVoisins */

        } // end if indDomCharge != -1
      } while (indDomCharge != -1);

      indDomVois = -1; /* to inform that we are stopping, indDomCharge == -1 and indDomVois == -1 */
      double DeltaNul = 0.0;
      PackEquil(InfoProc, indDomCharge, indDomVois, DeltaNul, TabTMP, TailleTMP, InfoProc->Split_Comm);

      TabTMP = DiffuseMessage(InfoProc, 0, TabTMP, TailleTMP);

      free((void*)FiltreDomaine);
      FiltreDomaine = NULL;

#ifdef ARCANE_DEBUG
      info() << " NbAppelsAEquil2Dom = " << NbAppelsAEquil2Dom;
#endif

    } /* end if me == 0 */
    else {
      double Delta;

      do {
        TabTMP = DiffuseMessage(InfoProc, 0, TabTMP, TailleTMP);
        UnpackEquil(TabTMP, TailleTMP, InfoProc->Split_Comm, &indDomCharge, &indDomVois, &Delta);

        if (indDomCharge != -1) {
          // Change the mark for each call
          marqueVu += 1;

          // Balancing between 2 domains
          Equil2Dom(MasqueDesNoeuds, MasqueDesElements, marqueVu, marqueNonVu,
                    InfoProc, Domaine, Maillage, indDomCharge, indDomVois, Delta);
          NbAppelsAEquil2Dom += 1;
        }

      } while (indDomCharge != -1);

    } /* end else if me == 0 */

    // Synchronization in the case where there have been modifications
    if (NbAppelsAEquil2Dom) {
      // Release of nodes in the interfaces
      LibereDomaine(Domaine);

#ifdef ARCANE_DEBUG
      info() << "cells_new_owner.synchronize() et changeOwnersFromCells()";
#endif
      // We only perform synchronization for a single series of calls to Equil2Dom
      VariableItemInt32& cells_new_owner = mesh->itemsNewOwner(IK_Cell);
      cells_new_owner.synchronize();
      changeOwnersFromCells();
      // Performs the effective transfer of data from one proc to another, without data compaction
      bool compact = mesh->properties()->getBool("compact");
      mesh->properties()->setBool("compact", false);
      mesh->exchangeItems();
      mesh->properties()->setBool("compact", compact);

      // Domain Update
      MAJDomaine(Domaine);

      /* We update the Maillage->ListeDomaines structure */
      MAJMaillageMaitre(InfoProc, Domaine, Maillage);
    } // end if NbAppelsAEquil2Dom

    free((void*)MasqueDesNoeuds);
    MasqueDesNoeuds = NULL;
    free((void*)MasqueDesElements);
    MasqueDesElements = NULL;

    AfficheEquilMaillage(Maillage);

    // Makes the domains connected as much as possible according to a tolerance
    // (we accept non-connected parts provided their size is > tol*average_size)
    marqueVu += 1;
    ConnexifieDomaine(InfoProc, Domaine, Maillage, tolConnexite);

    AfficheEquilMaillage(Maillage);

  } /* end while (iterEquilibrage<NbMaxIterEquil && NbAppelsAEquil2Dom!=0) */

  free(TabTMP);
  TabTMP = nullptr;

  debug() << " number of iterations for balancing = " << iterEquilibrage << " / " << NbMaxIterEquil << " max ";
  debug() << " = we are exiting SplitSDMeshPartitioner::Equilibrage";
  debug() << " -------------------------------------";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SplitSDMeshPartitioner::
Equil2Dom(int* MasqueDesNoeuds, int* MasqueDesElements, int marqueVu, int marqueNonVu,
          StrucInfoProc* InfoProc, StructureBlocEtendu* Domaine, StrucMaillage* Maillage,
          int indDomCharge, int indDomVois, double Delta)
{
  ARCANE_UNUSED(Maillage);

  debug() << "    we are entering Equil2Dom (indDomCharge:" << indDomCharge << ", indDomVois:" << indDomVois << ", Delta:" << Delta;

  // array of elements selected to be moved
  Arcane::UniqueArray<Arcane::Cell> ListeElements;

  if (InfoProc->me == indDomCharge) {
    int iIntf = 0;
    /* Does the interface between the two domains still exist? */
    for (iIntf = 0;
         iIntf < Domaine->NbIntf && Domaine->Intf[iIntf].NoDomVois != indDomVois;
         iIntf++) {
    }

    if (iIntf == Domaine->NbIntf) {
#if defined(ARCANE_DEBUG) || defined(DEBUG_PARTITIONER)
      pinfo() << "### the interface has disappeared ### between " << indDomCharge << " and " << indDomVois;
#endif
    }
    else {
      // selection of elements by a frontal traversal from the interface between the 2 sub-domains.
      SelectElements(MasqueDesNoeuds, MasqueDesElements,
                     marqueVu, marqueNonVu,
                     InfoProc, Domaine, Delta, indDomVois, ListeElements);
    }

    // marking for arcane data transfer
    IPrimaryMesh* mesh = this->mesh()->toPrimaryMesh();
    VariableItemInt32& cells_new_owner = mesh->itemsNewOwner(IK_Cell);

    for (int i = 0; i < ListeElements.size(); i++) {
      const Cell item = ListeElements[i];
      cells_new_owner[item] = indDomVois;
    }

  } // end if me == indDomCharge
#ifdef ARCANE_DEBUG
  else {
    info() << "SelectElements and other operations on processors " << indDomCharge << " and " << indDomVois;
  }

  info() << "     we are exiting    Equil2Dom";
#endif
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
void SplitSDMeshPartitioner::
SelectElements(int* MasqueDesNoeuds, int* MasqueDesElements, int marqueVu, int marqueNonVu,
               StrucInfoProc* InfoProc, StructureBlocEtendu* Domaine,
               double Delta, int indDomVois, Arcane::Array<Arcane::Cell>& ListeElements)
{
#ifdef ARCANE_DEBUG
  info() << "SplitSDMeshPartitioner::SelectElements(Domaine, Delta = " << Delta << ", indDomVois = " << indDomVois << ")";
  info() << " Domaine->NbElements = " << Domaine->NbElements;
  info() << " Domaine->PoidsDom = " << Domaine->PoidsDom;
#endif

  if (Delta <= 0.0) {
    perror() << "Delta <= 0 !!!";
  }

  IMesh* mesh = this->mesh();

  if (Delta >= Domaine->PoidsDom) {
#ifdef ARCANE_DEBUG
    pinfo() << " All the domain is selected on domain " << subDomain()->subDomainId()
            << ", with SelectElements, PoidsDom = " << Domaine->PoidsDom
            << ", Delta = " << Delta;
#endif
    // case where the entire domain is requested
    // this processor will end up empty! unless there is a transfer from another proc that compensates for the loss
    CellGroup own_cells = mesh->ownCells();
    ENUMERATE_CELL (i_item, own_cells) {
      const Cell item = *i_item;
      //MasqueDesElements[item.localId()] = marqueVu;
      ListeElements.add(item);
    }
  }
  else {
    // case where a subset must be selected

    int iIntf; /* interface index for neighbor indDomVois */
    int NbFrontsMax;
    int NbFronts;
    int* IndFrontsNoeuds;
    int* IndFrontsElements;

    for (iIntf = 0; iIntf < Domaine->NbIntf && Domaine->Intf[iIntf].NoDomVois != indDomVois; iIntf++) {
    }

    if (iIntf == Domaine->NbIntf) {
      pfatal() << " SelectElements cannot find the interface among the neighbors !!!";
    }

    NbFrontsMax = Domaine->NbElements / 2;

    /* nodes taken in the fronts */
    Arcane::UniqueArray<Arcane::Node> FrontsNoeuds;

    /* elements taken in the fronts */
    Arcane::Array<Arcane::Cell>& FrontsElements(ListeElements);

    IndFrontsNoeuds = (int*)malloc((size_t)(NbFrontsMax + 1) * sizeof(int));
    CHECK_IF_NOT_NULL(IndFrontsNoeuds);

    IndFrontsElements = (int*)malloc((size_t)(NbFrontsMax + 1) * sizeof(int));
    CHECK_IF_NOT_NULL(IndFrontsElements);

    /* starting front for the traversal */
    NbFronts = 1;

    /* we use the list of nodes in the interface as the starting front */
    for (int i = 0; i < Domaine->Intf[iIntf].ListeNoeuds.size(); i++)
      FrontsNoeuds.add(Domaine->Intf[iIntf].ListeNoeuds[i]);

    IndFrontsNoeuds[0] = 0;
    IndFrontsNoeuds[1] = Domaine->Intf[iIntf].ListeNoeuds.size();
    IndFrontsElements[0] = 0;
    IndFrontsElements[1] = 0;

    // we mark the phantom cells as already seen
    int me = subDomain()->subDomainId();
    CellGroup all_cells = mesh->allCells(); // elements on this processor (including phantom cells)
    ENUMERATE_CELL (i_item, all_cells) {
      const Cell cell = *i_item;
      if (cell.owner() != me)
        MasqueDesElements[cell.localId()] = marqueVu;
    }

    // we mark the already selected elements
    for (int i = 0; i < ListeElements.size(); i++) {
      const Cell cell = ListeElements[i];
      MasqueDesElements[cell.localId()] = marqueVu;
    }

    int retPF;
    // Frontal Traversal for a requested Delta weight
    retPF = ParcoursFrontalDelta(MasqueDesNoeuds, MasqueDesElements,
                                 marqueVu, marqueNonVu,
                                 Delta,
                                 &NbFronts, NbFrontsMax,
                                 FrontsNoeuds, IndFrontsNoeuds,
                                 FrontsElements, IndFrontsElements);

    /* the last front is an incomplete front, we concatenate it to the second to last for smoothing */
    if (NbFronts > 1) {
      IndFrontsNoeuds[NbFronts - 1] = IndFrontsNoeuds[NbFronts];
      IndFrontsElements[NbFronts - 1] = IndFrontsElements[NbFronts];
      NbFronts -= 1;
    }

    /* we perform front smoothing in the case where we haven't selected the entire domain
       and the Frontal Traversal proceeded correctly (is not blocked) */
    if (FrontsNoeuds.size() < Domaine->NbElements && retPF == 0)
      LissageDuFront(MasqueDesNoeuds, MasqueDesElements,
                     marqueVu, marqueNonVu,
                     NbFronts,
                     FrontsNoeuds, IndFrontsNoeuds,
                     FrontsElements, IndFrontsElements);

    free((void*)IndFrontsNoeuds);
    IndFrontsNoeuds = NULL;
    free((void*)IndFrontsElements);
    IndFrontsElements = NULL;
  }

  info() << " exiting: ListeElements.size() = " << ListeElements.size();
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
int SplitSDMeshPartitioner::
ParcoursFrontalDelta(int* MasqueDesNoeuds, int* MasqueDesElements,
                     int marqueVu, int marqueNonVu,
                     double Delta,
                     int* pNbFronts, int NbFrontsMax,
                     Arcane::Array<Arcane::Node>& FrontsNoeuds, int* IndFrontsNoeuds,
                     Arcane::Array<Arcane::Cell>& FrontsElements, int* IndFrontsElements)
{
#ifdef ARCANE_DEBUG
  info() << "       = we are entering FrontalTraversal :   (NbFronts = " << *pNbFronts << ", NbFrontsMax = " << NbFrontsMax << ")";
  info() << "  FrontsNoeuds.size() = " << FrontsNoeuds.size();
  info() << "  FrontsElements.size() = " << FrontsElements.size();
#endif

  int IndFn = 0; /* indices on the [Ind]FrontsNoeuds arrays */
  int IndFe = 0; /* indices on the [Ind]FrontsElements arrays */
  double PoidsActuel = 0.0;
  bool bloque = false;

  /* we mark the nodes and elements already in the fronts (to avoid duplicates) */
  for (IndFn = 0; IndFn < IndFrontsNoeuds[*pNbFronts]; IndFn++) {
    MasqueDesNoeuds[FrontsNoeuds[IndFn].localId()] = marqueVu;
  }

  for (IndFe = 0; IndFe < IndFrontsElements[*pNbFronts]; IndFe++) {
    MasqueDesElements[FrontsElements[IndFe].localId()] = marqueVu;
  }

  /* we put the nodes linked to the elements of this same front into the initial front
     if it hasn't been done */
  if (IndFrontsElements[*pNbFronts] > 0 && IndFrontsNoeuds[*pNbFronts] == 0) {
    //info()<<" Initialization from "<<IndFrontsElements[*pNbFronts]<<" elements";
    for (int ielm = 0; ielm < IndFrontsElements[*pNbFronts]; ielm++) {
      const Cell cell = FrontsElements[ielm];
      PoidsActuel += m_poids_aux_mailles[cell];

      for (int iepn = 0; iepn < cell.nbNode(); iepn++) {
        const Node nodeVois = cell.node(iepn); // neighbor node  // int NeighborNode

        /* for each new node, we insert it into the new front*/
        if (MasqueDesNoeuds[nodeVois.localId()] == marqueNonVu) {
          FrontsNoeuds.add(nodeVois);
          IndFn += 1;
          MasqueDesNoeuds[nodeVois.localId()] = marqueVu;
        }
      } /* end for iepn */
    }
    /* we set the nodes in the last existing front */
    IndFrontsNoeuds[*pNbFronts] = IndFn;
  }

  /*----------------------------------------------------------------*/
  /* loop until we have seen enough nodes or fronts */
  /*----------------------------------------------------------------*/
  do {
    /* for each node in the previous front, we look at the elements it possesses */
    for (int in = IndFrontsNoeuds[*pNbFronts - 1];
         in < IndFrontsNoeuds[*pNbFronts] && PoidsActuel < Delta;
         in++) {
      const Node node = FrontsNoeuds[in]; // node of the previous front  // int Node

      /* we avoid stopping without having all linked elements,
	 we risk having an element connected by only one node! */
      for (int inpe = 0; inpe < node.nbCell(); inpe++) {
        const Cell cell = node.cell(inpe); // element linked to this node  //int Element

        /* for this new element, we insert it into the new front
	   and look at the nodes it possesses */
        if (MasqueDesElements[cell.localId()] == marqueNonVu) {
          FrontsElements.add(cell);
          IndFe += 1;
          MasqueDesElements[cell.localId()] = marqueVu;

          PoidsActuel += m_poids_aux_mailles[cell];

          for (int iepn = 0; iepn < cell.nbNode(); iepn++) {
            const Node nodeVois = cell.node(iepn); // neighbor node  // int NeighborNode

            /* for each new node, we insert it into the new front*/
            if (MasqueDesNoeuds[nodeVois.localId()] == marqueNonVu) {
              FrontsNoeuds.add(nodeVois);
              IndFn += 1;
              MasqueDesNoeuds[nodeVois.localId()] = marqueVu;
            } /* end if MasqueDesNoeuds == marqueNonVu */
          } /* end for iepn */
        } /* end if MasqueDesElements == marqueNonVu */
      } /* end for inpe */
    } /* end for in */

    /* test the case where it would get blocked */
    if (IndFrontsNoeuds[*pNbFronts - 1] == IndFrontsNoeuds[*pNbFronts]) {
      bloque = true;
    }

    *pNbFronts += 1;
    IndFrontsNoeuds[*pNbFronts] = IndFn;
    IndFrontsElements[*pNbFronts] = IndFe;

  } while (*pNbFronts < NbFrontsMax && PoidsActuel < Delta && !bloque);

#ifdef ARCANE_DEBUG
  info() << " NbFronts = " << *pNbFronts;
  info() << " Delta = " << Delta;
  info() << " PoidsActuel = " << PoidsActuel;
  info() << " NbNoeuds   = " << IndFrontsNoeuds[*pNbFronts];
  info() << " NbElements = " << IndFrontsElements[*pNbFronts];
  info() << " blocked = " << (bloque ? "TRUE" : "FALSE");

  if (!(*pNbFronts < NbFrontsMax)) {
    info() << "       =  we stop after obtaining the maximum number of fronts " << NbFrontsMax;
  }
  else if (!(PoidsActuel < Delta)) {
    info() << "       =  we stop after obtaining the desired weight " << PoidsActuel;
  }
  else if (bloque) {
    info() << "       =  we are blocked (not connected ?) =";
  }
  else {
    info() << "       =  we stop because we have seen everything =";
  }
  info() << "       = we are exiting    FrontalTraversal  .   =";
#endif

  return ((bloque) ? 1 : 0);
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SplitSDMeshPartitioner::
LissageDuFront(int* MasqueDesNoeuds, int* MasqueDesElements,
               int marqueVu, int marqueNonVu,
               int NbFronts,
               Arcane::Array<Arcane::Node>& FrontsNoeuds, int* IndFrontsNoeuds,
               Arcane::Array<Arcane::Cell>& FrontsElements, int* IndFrontsElements)
{
  ARCANE_UNUSED(IndFrontsElements);

  debug() << "       we are entering LissageDuFront :    NbFronts = " << NbFronts;

  int NbElementsAjoutes = 0;

  Arcane::UniqueArray<Arcane::Cell> ElementsALiberer;

  /* Retrieval of elements taken in the last node front */
  for (int IndFn = IndFrontsNoeuds[NbFronts - 1]; IndFn < IndFrontsNoeuds[NbFronts]; IndFn++) {
    const Node node = FrontsNoeuds[IndFn]; // node of the previous front  // int Node

    for (int inpe = 0; inpe < node.nbCell(); inpe++) {
      const Cell cell = node.cell(inpe); //element linked to this node  //int Element

      /* if this element is unseen */
      if (MasqueDesElements[cell.localId()] == marqueNonVu) {
        /* we check if all nodes of this element are marked as seen */
        int iepn;
        for (iepn = 0; iepn < cell.nbNode() && MasqueDesNoeuds[cell.node(iepn).localId()] == marqueVu; iepn++) {
        }
        /* case where all nodes are marked */
        if (iepn == cell.nbNode()) {
          /* we add the element to the last front */

          FrontsElements.add(cell);
          NbElementsAjoutes += 1;
          MasqueDesElements[cell.localId()] = marqueVu;
        }
        else {
          /* we mark the element */
          MasqueDesElements[cell.localId()] = marqueVu; // so as not to pick it up immediately
          ElementsALiberer.add(cell);
        }

      } /* end if Element unseen */
    } /* end for inpe ... */
  } /* end for IndFn ... */

  // we reset those we marked back to unseen just so we only see them once
  for (int i = 0; i < ElementsALiberer.size(); i++)
    MasqueDesElements[ElementsALiberer[i].localId()] = marqueNonVu;

#ifdef ARCANE_DEBUG
  info() << "       we exit LissageDuFront (" << NbElementsAjoutes << " elements added)";
#endif
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
void SplitSDMeshPartitioner::
ConnexifieDomaine(StrucInfoProc* InfoProc, StructureBlocEtendu* Domaine, StrucMaillage* Maillage,
                  double tolConnexite)
{
#ifdef ARCANE_DEBUG
  info() << "    entering ConnexifieDomaine, tolConnexite = " << tolConnexite;
#endif

  int* MasqueDesNoeuds = GetMasqueDesNoeuds(InfoProc);
  int* MasqueDesElements = GetMasqueDesElements(InfoProc);
  int marqueVu = 1;
  int marqueNonVu = 0;

  // Maillage->NbElements is only known on the master!
  //   double tailleMoy = (double)Maillage->NbElements / (double)Maillage->NbDomainesMax;
  double tailleMoy = (double)Domaine->NbElements;

  // we mark the phantom cells as already seen
  int me = InfoProc->me;
  IPrimaryMesh* mesh = this->mesh()->toPrimaryMesh();
  CellGroup all_cells = mesh->allCells(); // elements on this processor (including phantom cells)

  ENUMERATE_CELL (i_item, all_cells) {
    Cell cell = *i_item;
    if (cell.owner() != me)
      MasqueDesElements[cell.localId()] = marqueVu;
  } // end ENUMERATE_CELL

  int NbElementsVus = 0;
  int NbElementsAVoir = Domaine->NbElements;
#ifdef ARCANE_DEBUG
  info() << " NbElementsAVoir = " << NbElementsAVoir;
#endif

  int NbFrontsMax;
  int NbFronts;
  int* IndFrontsNoeuds;
  int* IndFrontsElements;

  NbFrontsMax = Domaine->NbElements / 2;

  /* nodes taken into the fronts */
  Arcane::UniqueArray<Arcane::Node> FrontsNoeuds;

  /* elements taken into the fronts */
  Arcane::SharedArray<Arcane::SharedArray<Arcane::Cell>> ListeFrontsElements;

  IndFrontsNoeuds = (int*)malloc((size_t)(NbFrontsMax + 1) * sizeof(int));
  CHECK_IF_NOT_NULL(IndFrontsNoeuds);

  IndFrontsElements = (int*)malloc((size_t)(NbFrontsMax + 1) * sizeof(int));
  CHECK_IF_NOT_NULL(IndFrontsElements);

  /* we loop to see all elements of the domain */
  while (NbElementsVus < NbElementsAVoir) {

    FrontsNoeuds.clear();
    Arcane::SharedArray<Arcane::Cell> FrontsElements;

    /* search for an unseen element (it is put in the first front) */
    //TODO to optimize?
    bool trouve = false;
    ENUMERATE_CELL (i_item, all_cells) {
      const Cell cell = *i_item;
      if (!trouve && MasqueDesElements[cell.localId()] == marqueNonVu) {
        FrontsElements.add(cell);
        trouve = true;
      }
    }
    if (!trouve) {
      pfatal() << "ConnexifieDomaine is blocked while searching for a starting element";
    }

    NbFronts = 1;
    IndFrontsNoeuds[0] = 0;
    IndFrontsNoeuds[1] = 0;
    IndFrontsElements[0] = 0;
    IndFrontsElements[1] = 1;

    /* search for the associated connected set of unseen elements */
    ParcoursFrontalDelta(MasqueDesNoeuds, MasqueDesElements,
                         marqueVu, marqueNonVu,
                         Domaine->PoidsDom,
                         &NbFronts, NbFrontsMax,
                         FrontsNoeuds, IndFrontsNoeuds,
                         FrontsElements, IndFrontsElements);

    /* we set aside this set of elements */
    ListeFrontsElements.add(FrontsElements);

    NbElementsVus += FrontsElements.size();
#ifdef ARCANE_DEBUG
    info() << "  NbElementsVus+=" << FrontsElements.size();
#endif

  } /* end while (NbElementsVus < NbElementsAVoir) */

  // number of transferred components
  int nbCCTransf = 0;

  /* if there is more than one connected component */
  if (ListeFrontsElements.size() > 1) {

#ifdef ARCANE_DEBUG
    info() << "    NbComposantesConnexes = " << ListeFrontsElements.size();
#endif

    // analysis of the number of domains below the threshold
    int nbDomEnDessous = 0;
    int plusGrosseCC = 0; // largest connected component

    int seuil = (int)(tailleMoy * tolConnexite);
#ifdef ARCANE_DEBUG
    info() << "  threshold = " << seuil;
#endif

    for (int i = 0; i < ListeFrontsElements.size(); i++) {
      Arcane::Array<Arcane::Cell>& FrontsElements = ListeFrontsElements[i];
      plusGrosseCC = MAX(plusGrosseCC, FrontsElements.size());
      if (FrontsElements.size() < seuil)
        nbDomEnDessous += 1;
    }

#ifdef ARCANE_DEBUG
    info() << "  nbDomEnDessous = " << nbDomEnDessous;
#endif

    // to avoid taking all components
    if (nbDomEnDessous == ListeFrontsElements.size()) {
#ifdef ARCANE_DEBUG
      info() << " threshold lowered to " << plusGrosseCC;
#endif
      seuil = plusGrosseCC;
    }

    // for each component smaller than the threshold, we transfer it to a neighbor
    VariableItemInt32& cells_new_owner = mesh->itemsNewOwner(IK_Cell);
    for (int i = 0; i < ListeFrontsElements.size(); i++) {
      Arcane::Array<Arcane::Cell>& FrontsElements = ListeFrontsElements[i];
      if (FrontsElements.size() < seuil) {
        nbCCTransf += 1;

        // search for the neighboring subdomain having the maximum shared face
        int indDomVois = getDomVoisMaxFace(FrontsElements, me);

        // we assign the cells to this neighbor
        for (int j = 0; j < FrontsElements.size(); j++) {
          const Cell item = FrontsElements[j];
          cells_new_owner[item] = indDomVois;
        }

      } // if FrontsElements.size()<seuil
    } // end for i<ListeFrontsElements.size()
#ifdef ARCANE_DEBUG
    info() << " Number of transferred components: " << nbCCTransf;
#endif

  } // end if ListeFrontsElements.size() > 1

  /* memory deallocations */
  free((void*)IndFrontsNoeuds);
  IndFrontsNoeuds = NULL;
  free((void*)IndFrontsElements);
  IndFrontsElements = NULL;

  free((void*)MasqueDesNoeuds);
  MasqueDesNoeuds = NULL;
  free((void*)MasqueDesElements);
  MasqueDesElements = NULL;

  // we only synchronize if one of the subdomains made a modification
  // sum up nbCCTransf across all procs

  int nbCCTransfMin = 0;
  int nbCCTransfMax = 0;
  int nbCCTransfSum = 0;
  Int32 procMin = 0;
  Int32 procMax = 0;

  ISubDomain* sd = subDomain();
  IParallelMng* pm = sd->parallelMng();

  pm->computeMinMaxSum(nbCCTransf, nbCCTransfMin, nbCCTransfMax, nbCCTransfSum, procMin, procMax);

  bool synchroNecessaire = (nbCCTransfSum > 0);
#if defined(ARCANE_DEBUG) || defined(DEBUG_PARTITIONER)
  info() << " ConnexifieDomaine: nbCCTransfSum = " << nbCCTransfSum;
#endif

  if (synchroNecessaire) {
#ifdef ARCANE_DEBUG
    info() << "    we perform the synchronization";
#endif
    // we only synchronize for a single series of calls to Equil2Dom
    VariableItemInt32& cells_new_owner = mesh->itemsNewOwner(IK_Cell);
    cells_new_owner.synchronize();
    changeOwnersFromCells();
    // performs the effective transfer of data from one proc to another, without data compaction
    bool compact = mesh->properties()->getBool("compact");
    mesh->properties()->setBool("compact", false);
    mesh->exchangeItems();
    mesh->properties()->setBool("compact", compact);

    // Update Domain
    MAJDomaine(Domaine);

    /* we update the Maillage->ListeDomaines structure */
    MAJMaillageMaitre(InfoProc, Domaine, Maillage);
  }
#ifdef ARCANE_DEBUG
  info() << "    we exit ConnexifieDomaine";
#endif
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
int SplitSDMeshPartitioner::
getDomVoisMaxFace(Arcane::Array<Arcane::Cell>& ListeElements, int me)
{
  int indDomVois = -1;

  // number of faces per neighboring subdomain
  std::map<int, int> indVois_nbFace;

  for (int i = 0; i < ListeElements.size(); i++) {
    const Cell cell = ListeElements[i];

    for (int j = 0; j < cell.nbFace(); j++) {
      const Face face = cell.face(j);

      if (!face.isSubDomainBoundary()) {
        int id1 = face.backCell().owner();
        int id2 = face.frontCell().owner();

        if (id1 == me && id2 != me)
          indVois_nbFace[id2] += 1;
        else if (id1 != me && id2 == me)
          indVois_nbFace[id1] += 1;
      }
    }
  }

  // search for the largest number of faces
  int maxNbFaces = 0;
  std::map<int, int>::iterator iter;
  for (iter = indVois_nbFace.begin();
       iter != indVois_nbFace.end();
       ++iter) {
    int nbFaces = (*iter).second;
    if (nbFaces > maxNbFaces) {
      maxNbFaces = nbFaces;
      indDomVois = (*iter).first;
    }
  }

  if (indDomVois == -1)
    pfatal() << "indDomVois always at -1 !!!";

#ifdef ARCANE_DEBUG
  pinfo() << " getDomVoisMaxFace, me = " << me << ", ListeElements.size() = " << ListeElements.size()
          << ", indDomVois = " << indDomVois << ", maxNbFaces = " << maxNbFaces;
#endif
  return indDomVois;
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
int* SplitSDMeshPartitioner::
GetMasqueDesNoeuds(StrucInfoProc* InfoProc)
{
  int* MasqueDesNoeuds = NULL;

  IMesh* mesh = this->mesh();
  // search for the largest id
  int maxNodeLocalId = 0;

  NodeGroup all_nodes = mesh->allNodes();
  ENUMERATE_NODE (i_item, all_nodes) {
    const Node node = *i_item;
    maxNodeLocalId = MAX(maxNodeLocalId, node.localId());
  }
#ifdef ARCANE_DEBUG
  info() << "SplitSDMeshPartitioner::GetMasqueDesNoeuds(), maxNodeLocalId = " << maxNodeLocalId;
#endif

  /* allocation and initialization to 0 of the mask */
  MasqueDesNoeuds = (int*)calloc((size_t)(maxNodeLocalId + 1), sizeof(int));
  CHECK_IF_NOT_NULL(MasqueDesNoeuds);

  return MasqueDesNoeuds;
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
int* SplitSDMeshPartitioner::
GetMasqueDesElements(StrucInfoProc* InfoProc)
{
  int* MasqueDesElements = NULL;

  IMesh* mesh = this->mesh();
  // search for the largest id
  int maxCellLocalId = 0;

  CellGroup all_cells = mesh->allCells(); // elements on this processor (including phantom cells)
  ENUMERATE_CELL (i_item, all_cells) {
    const Cell cell = *i_item;
    maxCellLocalId = MAX(maxCellLocalId, cell.localId());
  }

#ifdef ARCANE_DEBUG
  info() << "SplitSDMeshPartitioner::GetMasqueDesElements(), maxCellLocalId = " << maxCellLocalId;
#endif

  /* allocation and initialization to 0 of the mask */
  MasqueDesElements = (int*)calloc((size_t)(maxCellLocalId + 1), sizeof(int));
  CHECK_IF_NOT_NULL(MasqueDesElements);

  return MasqueDesElements;
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SplitSDMeshPartitioner::
LibereInfoProc(StrucInfoProc*& InfoProc)
{
  ARCANE_UNUSED(InfoProc);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SplitSDMeshPartitioner::
LibereDomaine(StructureBlocEtendu*& Domaine)
{
#ifdef ARCANE_DEBUG
  info() << "LibereDomaine(...)";
#endif

  if (Domaine->NbIntf != 0) {
    if (Domaine->Intf != NULL) {
      for (int i = 0; i < Domaine->NbIntf; i++)
        Domaine->Intf[i].ListeNoeuds.clear();
      delete[] Domaine->Intf;
      Domaine->Intf = NULL;
    }
  }

  Domaine->NbIntf = 0;
  Domaine->NbElements = 0;
  Domaine->PoidsDom = 0.0;
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
void SplitSDMeshPartitioner::
LibereMaillage(StrucMaillage*& Maillage)
{
#ifdef ARCANE_DEBUG
  info() << "LibereMaillage(...)";
#endif

  if (Maillage->NbElements != 0) {
    if (Maillage->ListeDomaines != NULL) {
      for (int i = 0; i < Maillage->NbDomainesMax; i++) {
        if (Maillage->ListeDomaines[i].ListeVoisins != NULL) {
          delete[] Maillage->ListeDomaines[i].ListeVoisins;
          Maillage->ListeDomaines[i].ListeVoisins = NULL;
        }
      }
      delete[] Maillage->ListeDomaines;
      Maillage->ListeDomaines = NULL;
    }
  }
  Maillage->NbElements = 0;
  Maillage->Poids = 0.0;
  Maillage->NbDomainesPleins = 0;
  Maillage->NbProcsVides = 0;
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
void SplitSDMeshPartitioner::
AfficheDomaine(int NbDom, StructureBlocEtendu* Domaine)
{
  int i;
  int idom;

  for (idom = 0; idom < NbDom; idom++) {
    info() << " --------------------";
    info() << " --- Domaine (" << idom << ") ---";
    info() << " --------------------";

    if (Domaine == NULL) {
      info() << " Domaine vide !  (pointeur NULL)";
    }
    else if (Domaine[idom].NbElements == 0) {
      info() << " Domaine vide !  (pas d'éléments)";
    }
    else {
      info() << " NbElements = " << Domaine[idom].NbElements;
      info() << " PoidsDom   = " << Domaine[idom].PoidsDom;
      info() << " Interfaces (NbIntf = " << Domaine[idom].NbIntf << ") :";
      for (i = 0; i < Domaine[idom].NbIntf; i++) {
        info() << " (" << i << ") NoDomVois = " << Domaine[idom].Intf[i].NoDomVois
               << ", number of nodes " << Domaine[idom].Intf[i].ListeNoeuds.size();
      }

    } /* end else if 'empty domain' */
  } /* end for idom */
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
void SplitSDMeshPartitioner::
AfficheMaillage(StrucMaillage* Maillage)
{
  info() << " ----------------";
  info() << " ----- Mesh -----";
  info() << " ----------------";
  if (Maillage == NULL) {
    info() << " Mesh structure is empty!";
  }
  else {
    info() << " NbElements (total)   = " << Maillage->NbElements;
    info() << " Poids      (total)   = " << Maillage->Poids;
    info() << " NbDomainesMax        = " << Maillage->NbDomainesMax;
    info() << " NbDomainesPleins     = " << Maillage->NbDomainesPleins;
    info() << " NbProcsVides         = " << Maillage->NbProcsVides;

    if (Maillage->ListeDomaines == NULL) {
      info() << " Maillage.ListeDomaines == NULL";
    }
    else {
      AfficheListeDomaines(Maillage->ListeDomaines, Maillage->NbDomainesMax);
    }

  } /* else if Maillage==NULL */
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
void SplitSDMeshPartitioner::
AfficheListeDomaines(StrucListeDomMail* ListeDomaines, int NbDomaines)
{
  info() << "  ListeDomaines :";
  for (int i = 0; i < NbDomaines; i++) {
    info() << " (" << i << ") NbElements  = " << ListeDomaines[i].NbElements << "; Weight    = " << ListeDomaines[i].Poids;
    info() << " (" << i << ") NbVoisins = " << ListeDomaines[i].NbVoisins << "; ListeVoisins :";
    for (int j = 0; j < ListeDomaines[i].NbVoisins; j++) {
      info() << " (" << i << ")   NoDomVois  = " << ListeDomaines[i].ListeVoisins[j].NoDomVois
             << "; NbNoeudsInterface  = " << ListeDomaines[i].ListeVoisins[j].NbNoeudsInterface
             << "; Delta  = " << ListeDomaines[i].ListeVoisins[j].Delta;
    }
  }
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
void SplitSDMeshPartitioner::
AfficheEquilMaillage(StrucMaillage* Maillage)
{
  info() << " AfficheEquilMaillage(...)";

  double poidsMin = 0.0;
  double poidsMax = 0.0;

  if (Maillage->ListeDomaines != NULL) {

    poidsMin = Maillage->ListeDomaines[0].Poids;

    for (int i = 0; i < Maillage->NbDomainesMax; i++) {
      double poidsDom = Maillage->ListeDomaines[i].Poids;

      if (poidsDom > poidsMax)
        poidsMax = poidsDom;
      if (poidsDom < poidsMin)
        poidsMin = poidsDom;
    }

    info() << "   INFO balancing / nodes : max : " << poidsMax << ", min : " << poidsMin
           << ", max/avg = " << poidsMax / (Maillage->Poids / (double)Maillage->NbDomainesMax);
  }
  else {
    info() << "AfficheEquilMaillage : Maillage->ListeDomaines == NULL";
  }
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/**
  Receives an array of unknown size via MPI.
  Allocates this array and returns it (as output).

  {\em Note:} the array size is positive when known; otherwise, MPI\_Probe
  and MPI\_Get\_count functions are called to determine the size.

  @memo    Receiving an array using the MPI communication library.
  @param   InfoProc (I) structure describing the processor on which the application runs.
  @param   FromProc (I) number of the sending processor.
  @param   Tag      (I) tag to distinguish messages.
  @param   pTailleTMP (I/O) pointer to the size in bytes of the TabTMP array:
					  <=0 if the array size is unknown
					  > 0 if the array size is known.

  @return  (void*) TabTMP: the array that is sent.
  @see     MAJMaillageMaitre
  @author  Eric Brière de l'Isle, ONERA, DRIS/SRL
  @version Created September 1996 - 26/08/98
*/
void* SplitSDMeshPartitioner::RecoitMessage(StrucInfoProc* InfoProc, int FromProc, int Tag, int* pTailleTMP)
{
  void* TabTMP;
  int ierror; /* error return on MPI lib */
  MPI_Status status; /* for MPI communication  */

  if (*pTailleTMP <= 0) {
    /* to retrieve information about the size of the arrays that will follow,
	 wait for a message to arrive */
    ierror = MPI_Probe(FromProc,
                       Tag,
                       InfoProc->Split_Comm,
                       &status);

    if (ierror != MPI_SUCCESS) {
      InfoProc->m_service->pfatal() << "Problem on " << InfoProc->me << " communication from "
                                    << FromProc << ", during MPI_Probe";
    }

    /* retrieving the size of the message | array */
    ierror = MPI_Get_count(&status,
                           MPI_PACKED,
                           pTailleTMP);

    if (ierror != MPI_SUCCESS) {
      InfoProc->m_service->pfatal() << "Problem on " << InfoProc->me << " communication from "
                                    << FromProc << ", during MPI_Get_count";
    }
  } /* end if *pTailleTMP <= 0 */

  /* allocation for the buffer array */
  if (*pTailleTMP > 0) {
    TabTMP = malloc((size_t)*pTailleTMP);
    CHECK_IF_NOT_NULL(TabTMP);
  }
  else
    TabTMP = NULL;

  /* receiving the message */
  ierror = MPI_Recv(TabTMP,
                    *pTailleTMP,
                    MPI_PACKED,
                    FromProc,
                    Tag,
                    InfoProc->Split_Comm,
                    &status);

  if (ierror != MPI_SUCCESS) {
    InfoProc->m_service->pfatal() << "Problem on " << InfoProc->me << " communication from "
                                  << FromProc << ", during MPI_Recv";
  }

  return (TabTMP);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/**
  Function responsible for sending an array.

  {\em Note:} The array size is in bytes.

  @memo    Sending a message.
  @param   InfoProc  (I) structure describing the processor on which the application runs.
  @param   ToProc    (I) destination computing node.
  @param   Tag       (I) tag to differentiate messages.
  @param   TabTMP    (I) array that is sent.
  @param   TailleTMP (I) size of TabTMP (in bytes).
  @return  void
  @see     Equilibrage,MAJMaillageMaitre
  @author  Eric Brière de l'Isle, ONERA, DRIS/SRL
  @version Created January 1997 - 26/08/98
*/
void SplitSDMeshPartitioner::EnvoieMessage(StrucInfoProc* InfoProc, int ToProc, int Tag, void* TabTMP, int TailleTMP)
{
  int ierror; /* error return on MPI functions */

  ierror = MPI_Send((void*)TabTMP,
                    TailleTMP,
                    MPI_PACKED,
                    ToProc,
                    Tag,
                    InfoProc->Split_Comm);

  if (ierror != MPI_SUCCESS) {
    InfoProc->m_service->pfatal() << "Problem on " << InfoProc->me << " communication to "
                                  << ToProc << ", during MPI_Send";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// /**
//   Function responsible for sending an array with non-blocking communication.

//   {\em Note:} The array size is in bytes.

//   @memo    Sending a non-blocking message.
//   @param   InfoProc  (I) structure describing the processor on which the application runs.
//   @param   ToProc    (I) destination computing node.
//   @param   Tag       (I) tag to differentiate messages.
//   @param   TabTMP    (I) array that is sent.
//   @param   TailleTMP (I) size of TabTMP (in bytes).
//   @return  (MPI_Request*) prequest: pointer to an MPI-specific structure for message management.
//   @see
//   @author  Eric Brière de l'Isle, ONERA, DRIS/SRL
//   @version Created December 1998
// */
// MPI_Request* EnvoieIMessage(StrucInfoProc* InfoProc, int ToProc, int Tag, void* TabTMP, int TailleTMP)
// {
//   int ierror;
//   MPI_Request *prequest;

//   prequest = (MPI_Request*) malloc(sizeof(MPI_Request));
//   CHECK_IF_NOT_NULL(prequest);

//   ierror = MPI_Isend ((void *) TabTMP,
// 		      TailleTMP,
// 		      MPI_PACKED,
// 		      ToProc,
// 		      Tag,
// 		      InfoProc->Split_Comm,
// 		      prequest);

//   if (ierror != MPI_SUCCESS) {
//     InfoProc->m_service->pfatal()<<"Problem on "<<InfoProc->me<<" communication to "
// 				 <<ToProc<<", during MPI_Isend";
//   }

//   return prequest;

// }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/**
  Function responsible for broadcasting an array (sending and receiving).
  An MPI\_Bcast is used, but the array size must be known on all processors.

  {\em Note:} The array size is in bytes.

  @memo    Broadcasting a message.
  @param   InfoProc  (I) structure describing the processor on which the application runs.
  @param   FromProc  (I) computing node from which the array is broadcast.
  @param   TabTMP    (I) array that is sent.
  @param   TailleTMP (I) size of TabTMP (in bytes).
  @return  (void *) TabTMP: the array that is received.
  @see     Equilibrage
  @author  Eric Brière de l'Isle, ONERA, DRIS/SRL
  @version Created October 1998
*/
void* SplitSDMeshPartitioner::DiffuseMessage(StrucInfoProc* InfoProc, int FromProc, void* TabTMP, int TailleTMP)
{
  int ierror; /* error return on MPI functions */

  if (TailleTMP <= 0) {
    InfoProc->m_service->pfatal() << "DiffuseMessage from " << InfoProc->me << ", it is necessary that the array size is known !!!\n";
  }

  if (InfoProc->me != FromProc) {
    if (TabTMP == NULL) {
      TabTMP = malloc((size_t)TailleTMP);
      CHECK_IF_NOT_NULL(TabTMP);
    }
  } /* end if me != FromProc */

  /* broadcasting the TabTMP array */
  ierror = MPI_Bcast(TabTMP,
                     TailleTMP,
                     MPI_PACKED,
                     FromProc,
                     InfoProc->Split_Comm);

  if (ierror != MPI_SUCCESS) {
    InfoProc->m_service->pfatal() << "Problem on " << InfoProc->me << " from "
                                  << FromProc << ", during MPI_Bcast";
  }

  return (TabTMP);
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \brief Gives the size in bytes required for the memory storage of a domain
  (without the list of interface nodes) for communication purposes.

  @memo    Gives the size in bytes of a StructureBlocEtendu
  @param   Domaine (I) StructureBlocEtendu
  @return  (int) : the size in bytes
  @see     PackDom, UnpackDom
  @author  Eric Brière de l'Isle, November 2007
*/
Integer SplitSDMeshPartitioner::
TailleDom(StructureBlocEtendu* Domaine)
{
  size_t s = (2 + 2 * Domaine->NbIntf) * sizeof(int) + sizeof(double);
  return arcaneCheckArraySize(s);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/**
  Stores the content of a Domain in an array for communication (without the list of interface nodes).

  @memo    Storing a StructureBlocEtendu in an array.
  @param   Domaine (I) StructureBlocEtendu
  @param   TabTMP (I/O) array that is filled
  @param   TailleTMP (I) total size of the TabTMP array
  @param   comm (I) communication environment
  @return  void
  @see     TailleDom, UnpackDom
  @author  Eric Brière de l'Isle, November 2007
*/
void SplitSDMeshPartitioner::
PackDom(StrucInfoProc* InfoProc, StructureBlocEtendu* Domaine, void* TabTMP,
        int TailleTMP, MPI_Comm comm)
{
  int position = 0; // initialization to put into the array at the beginning
  int ier;

  ier = MPI_Pack(&Domaine->NbElements, 1, MPI_INT, TabTMP, TailleTMP, &position, comm);
  CHECK_MPI_PACK_ERR(ier);

  ier = MPI_Pack(&Domaine->PoidsDom, 1, MPI_DOUBLE, TabTMP, TailleTMP, &position, comm);
  CHECK_MPI_PACK_ERR(ier);

  ier = MPI_Pack(&Domaine->NbIntf, 1, MPI_INT, TabTMP, TailleTMP, &position, comm);
  CHECK_MPI_PACK_ERR(ier);

  for (int i = 0; i < Domaine->NbIntf; i++) {
    ier = MPI_Pack(&Domaine->Intf[i].NoDomVois, 1, MPI_INT, TabTMP, TailleTMP, &position, comm);
    CHECK_MPI_PACK_ERR(ier);

    int NbNoeuds = Domaine->Intf[i].ListeNoeuds.size();
    ier = MPI_Pack(&NbNoeuds, 1, MPI_INT, TabTMP, TailleTMP, &position, comm);
    CHECK_MPI_PACK_ERR(ier);
  }
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/**
  Extracts a Domain from an array received via communication.

  @memo    Extracting data to populate a StrucListeDomMail
  @param   TabTMP (I) array from which the information is extracted
  @param   TailleTMP (I) total size of the TabTMP array
  @param   comm (I) communication environment for the library
  @param   DomMail (I/O) StrucListeDomMail
  @return  void
  @see     TailleDom, PackDom
  @author  Eric Brière de l'Isle, November 2007
*/
void SplitSDMeshPartitioner::UnpackDom(void* TabTMP, int TailleTMP, MPI_Comm comm, StrucListeDomMail* DomMail)
{
  int position = 0; // initialization to take from the beginning of the array

  MPI_Unpack(TabTMP, TailleTMP, &position, &DomMail->NbElements, 1, MPI_INT, comm);
  MPI_Unpack(TabTMP, TailleTMP, &position, &DomMail->Poids, 1, MPI_DOUBLE, comm);
  MPI_Unpack(TabTMP, TailleTMP, &position, &DomMail->NbVoisins, 1, MPI_INT, comm);

  DomMail->ListeVoisins = new StrucListeVoisMail[DomMail->NbVoisins];

  for (int i = 0; i < DomMail->NbVoisins; i++) {
    MPI_Unpack(TabTMP, TailleTMP, &position, &DomMail->ListeVoisins[i].NoDomVois, 1, MPI_INT, comm);
    MPI_Unpack(TabTMP, TailleTMP, &position, &DomMail->ListeVoisins[i].NbNoeudsInterface, 1, MPI_INT, comm);
    DomMail->ListeVoisins[i].Delta = 0.0;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/**
  Gives the size in bytes required for the memory storage of an integer for the loaded domain, another integer for the neighbor domain, and a double for Delta (weight transfer).
*/

int SplitSDMeshPartitioner::TailleEquil()
{
  return 2 * sizeof(int) + sizeof(double);
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/**
  Stores the 2 integers and the double
*/
void SplitSDMeshPartitioner::PackEquil(StrucInfoProc* InfoProc, int indDomCharge, int indDomVois, double Delta, void* TabTMP, int TailleTMP, MPI_Comm comm)
{
  int position = 0; // initialization to put into the array at the beginning
  int ier;
  ier = MPI_Pack(&indDomCharge, 1, MPI_INT, TabTMP, TailleTMP, &position, comm);
  CHECK_MPI_PACK_ERR(ier);

  ier = MPI_Pack(&indDomVois, 1, MPI_INT, TabTMP, TailleTMP, &position, comm);
  CHECK_MPI_PACK_ERR(ier);

  ier = MPI_Pack(&Delta, 1, MPI_DOUBLE, TabTMP, TailleTMP, &position, comm);
  CHECK_MPI_PACK_ERR(ier);
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/**
  Extracts the 2 integers and the double from an array received via communication.
*/
void SplitSDMeshPartitioner::UnpackEquil(void* TabTMP, int TailleTMP, MPI_Comm comm, int* indDomCharge, int* indDomVois, double* Delta)
{
  int position = 0; // initialization to take from the beginning of the array
  MPI_Unpack(TabTMP, TailleTMP, &position, indDomCharge, 1, MPI_INT, comm);
  MPI_Unpack(TabTMP, TailleTMP, &position, indDomVois, 1, MPI_INT, comm);
  MPI_Unpack(TabTMP, TailleTMP, &position, Delta, 1, MPI_DOUBLE, comm);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(SplitSDMeshPartitioner,
                        ServiceProperty("SplitSD", ST_SubDomain),
                        ARCANE_SERVICE_INTERFACE(IMeshPartitioner),
                        ARCANE_SERVICE_INTERFACE(IMeshPartitionerBase));

ARCANE_REGISTER_SERVICE_SPLITSDMESHPARTITIONER(SplitSD, SplitSDMeshPartitioner);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
