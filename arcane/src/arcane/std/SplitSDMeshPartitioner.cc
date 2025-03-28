// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SplitSDMeshPartitioner.cc
   
   Partitioneur de maillage reprenant le fonctionnement (simplifié) de SplitSD 
   utilisé à Dassault Aviation et développé à l'ONERA par EB en 1996-99     
*/
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
, m_poids_aux_mailles(VariableBuildInfo(sbi.mesh(),"MeshPartitionerCellsWeight",IVariable::PNoDump|IVariable::PNoRestore))
{
  info() << "SplitSDMeshPartitioner::SplitSDMeshPartitioner(...)";  
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SplitSDMeshPartitioner::
partitionMesh(bool initial_partition)
{
  info() << "Equilibrage de charge avec SplitSDMeshPartitioner";

  // on ne supporte pas les contraintes
  // car on utilise le maillage Arcane pour les parcours frontaux, 
  // il n'est donc pas prévu pour le moment de tenir compte des groupes 
  // de mailles associées aux contraintes
  _initArrayCellsWithConstraints();
  if (haveConstraints())
    throw FatalErrorException("SplitSDMeshPartitioner: On ne supporte pas les contraintes avec SplitSD");

  // initialisation des structures internes

  StrucInfoProc*       InfoProc = NULL;/* structure décrivant le processeur sur lequel tourne l'application */
  StructureBlocEtendu*  Domaine = NULL;/* structure décrivant sommairement le bloc, c.a.d. la partie de la topologie localisée sur ce processeur */
  StrucMaillage*       Maillage = NULL;/* structure décrivant sommairement le maillage global */

  // initialisation des données propres au partitionneur, m_cells_weight doit être calculé
  init(initial_partition, InfoProc, Domaine, Maillage);

  // processus itératif de rééquilibrage de charge par déplacement d'éléments d'un sous-domaine à l'autre
  Equilibrage(InfoProc, Domaine, Maillage); 
  
  // on vide les structures
  fin(InfoProc, Domaine, Maillage);
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
void SplitSDMeshPartitioner::
init(bool initial_partition, StrucInfoProc* &InfoProc, StructureBlocEtendu* &Domaine, StrucMaillage* &Maillage)
{
  info() << "SplitSDMeshPartitioner::init("<<initial_partition<<",...)";

  ISubDomain*   sd = subDomain();
  IParallelMng* pm = sd->parallelMng();

  /* Initialisation memoire de InfoProc  */
  InfoProc = new StrucInfoProc();

  InfoProc->me = sd->subDomainId();
  InfoProc->nbSubDomain = pm->commSize();
  InfoProc->m_service = this;
  // MPI_Comm_dup(MPI_COMM_WORLD,&InfoProc->Split_Comm); /* Attribution d'un communicateur propre à notre partitioneur */
  InfoProc->Split_Comm = *(MPI_Comm*)getCommunicator();
  initConstraints();

  // initialisation des poids aux mailles
  initPoids(initial_partition);
#if 0
  if (!initial_partition){
    info() << "Initialize new owners";
    // Initialise le new owner pour le cas où il n'y a pas besoin de rééquilibrer
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

  /* Initialisation memoire de Domaine */
  Domaine = new StructureBlocEtendu();
  
  Domaine->NbIntf     = 0;
  Domaine->Intf       = NULL;
  Domaine->NbElements = 0;
  Domaine->PoidsDom   = 0.0;

  MAJDomaine(Domaine);

  /* Initialisation memoire de Maillage */
  Maillage = new StrucMaillage();

  Maillage->NbElements       = 0;
  Maillage->NbDomainesMax    = pm->commSize();
  Maillage->NbDomainesPleins = 0;
  Maillage->ListeDomaines    = NULL;
  Maillage->NbProcsVides     = 0;
  Maillage->Poids            = 0.0;

  // on ne met à jour que sur le processeur maitre (le 0) la structure, en fonction de Domaine
  MAJMaillageMaitre(InfoProc, Domaine, Maillage);
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
void SplitSDMeshPartitioner::
initPoids(bool initial_partition)
{
  ARCANE_UNUSED(initial_partition);

  /* récupération du poids aux mailles pour qu'il suivent les cellules */
  IMesh* mesh = this->mesh();
  CellGroup own_cells = mesh->ownCells();
//   if (initial_partition || m_cells_weight.empty())
//     ENUMERATE_CELL(iitem,own_cells){
//       const Cell& cell = *iitem
//       m_poids_aux_mailles[cell] = 1.0;
//     }
//   else{
//     Integer nb_weight = nbCellWeight();
  SharedArray<float> cell_weights = cellsWeightsWithConstraints(1, true); // 1 poids seulement
  ENUMERATE_CELL(iitem,own_cells){
    m_poids_aux_mailles[iitem] = cell_weights[iitem.index()];
  }
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
void SplitSDMeshPartitioner::
fin(StrucInfoProc* &InfoProc, StructureBlocEtendu* &Domaine, StrucMaillage* &Maillage)
{
  info() << "SplitSDMeshPartitioner::fin(...)";
  LibereInfoProc(InfoProc);
  LibereDomaine(Domaine);
  LibereMaillage(Maillage);

  delete InfoProc;
  delete Domaine ;
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
  debug() <<" ----------------------------------------";
  debug() << "SplitSDMeshPartitioner::MAJDomaine(...)";

  LibereDomaine(Domaine);

  int me = subDomain()->subDomainId();

  // remplissage de Domaine->Intf, listes de noeuds par sous-domaine voisin

  IMesh* mesh = this->mesh();
  FaceGroup all_faces = mesh->allFaces(); // faces sur ce processeur
  debug() << " all_faces.size() = =" <<all_faces.size();

  std::map<int, SharedArray<Face> > vois_faces; // liste des faces par sous-domaine voisin
 
  ENUMERATE_FACE(i_item,all_faces){
    const Face face = *i_item;
    if (!face.isSubDomainBoundary()){
      int id1 = face.backCell().owner();
      int id2 = face.frontCell().owner();
      
      if (id1 != id2){ // cas du voisinage avec un autre domaine
        int idv = -1; // numéro du voisin
        if (id1==me)
          idv = id2;
        else if (id2==me)
          idv = id1;
        else
          continue;
        // info() << "idv = "<<idv<< " pour la face " <<face.uniqueId();

        // on ajoute l'indice de la face pour un sous-domaine
        SharedArray<Face> &v_face = vois_faces[idv];
        v_face.add(face);

      } // end if (id1 != id2)
    } // end if (!face.isBoundary())
  } // end ENUMERATE_FACE

  UniqueArray<int> filtreNoeuds(mesh->nodeFamily()->maxLocalId());
  filtreNoeuds.fill(0); // initialisation du filtre
  int marque = 0;

  // pour chacun des sous-domaines voisins, on recherche les noeuds dans l'interface
  // en évitant les doublons à l'aide du filtre sur le noeuds

  Domaine->NbIntf = arcaneCheckArraySize(vois_faces.size());

  // les mailles sur ce proc (sans les mailles fantomes)
  Domaine->NbElements = mesh->ownCells().size();

  // calcul du poids total d'un domaine => utilisation m_poids_aux_mailles
  Domaine->PoidsDom = 0.0;
  CellGroup own_cells = mesh->ownCells();
  ENUMERATE_CELL(iitem,own_cells){
    Cell cell = *iitem;
    Domaine->PoidsDom += m_poids_aux_mailles[cell];
  }

#ifdef ARCANE_DEBUG
  info() << "Domaine->NbIntf = "<<Domaine->NbIntf;
  info() << "Domaine->NbElements = "<<Domaine->NbElements;
  info() << "Domaine->PoidsDom = "<<Domaine->PoidsDom;
#endif

  Domaine->Intf = new StructureInterface[Domaine->NbIntf];
  unsigned int ind = 0;
  for (std::map<int, SharedArray<Face> >::iterator iter_vois = vois_faces.begin();
       iter_vois!=vois_faces.end(); ++iter_vois) {

    marque+=1;

    Domaine->Intf[ind].NoDomVois = (*iter_vois).first;
    Array<Face> &v_face = (*iter_vois).second;

#ifdef ARCANE_DEBUG
    info() << "Domaine->Intf["<<ind<<"].NoDomVois = "<<Domaine->Intf[ind].NoDomVois;
    info() << " v_face.size() = "<<v_face.size();
#endif

    for (int i = 0; i<v_face.size(); i++){
      const Face& face = v_face[i];
//       info() << " v_face["<<i<<"].uniqueId() = " << face.uniqueId()
// 	     << ", backCell: "<<face.backCell().owner() <<", frontCell: "<< face.frontCell().owner();

      for( Integer z=0; z<face.nbNode(); ++z ){
	  const Node node = face.node(z);
	  Integer node_local_id = node.localId();
	  if (filtreNoeuds[node_local_id] != marque){
	    Domaine->Intf[ind].ListeNoeuds.add(node);
	    filtreNoeuds[node_local_id] = marque;
	  }
	}
      
    }// end for iter_face
#ifdef ARCANE_DEBUG
    info() << " ListeNoeuds.size() = "<<Domaine->Intf[ind].ListeNoeuds.size();
#endif

    ind+=1;
  } // end for iter_vois

#ifdef ARCANE_DEBUG
  info() << "MAJDomaine => ";
  AfficheDomaine(1,Domaine);
  info() <<" ----------------------------------------";
#endif
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
void SplitSDMeshPartitioner::
MAJMaillageMaitre(StrucInfoProc* InfoProc, StructureBlocEtendu* Domaine, StrucMaillage* Maillage)
{
  // prérequis: avoir fait un MAJDomaine avant de venir ici

#ifdef ARCANE_DEBUG
  info() << "SplitSDMeshPartitioner::MAJMaillageMaitre(...)";  
  LibereMaillage(Maillage);
#endif

  void* TabTMP;    /* tableau pour les communications Domaine => Maillage->ListeDomaines[*] */
  int   TailleTab; /* taille du tableau TabTMP (en octets) */
  int   iDom;      /* indice de boucle sur les domaines */

  /* on concentre l'information dans TabTMP */
  TailleTab = TailleDom(Domaine);
  TabTMP =  malloc ((size_t) TailleTab);
      
  PackDom(InfoProc, Domaine,  TabTMP, TailleTab, InfoProc->Split_Comm);
  

  /* on fait les envois vers le maitre, pour tous les domaines 
     (sauf le maitre qui fait juste une copie) */
  if (InfoProc->me == 0){
#ifdef DEBUG
    info()<<"   ***************************";
    info()<<"   * Avant MAJMaillageMaitre *";
    AfficheMaillage (Maillage);
    info()<<"   ***************************";
#endif

    if (Maillage->ListeDomaines == NULL){
      Maillage->ListeDomaines = new StrucListeDomMail[InfoProc->nbSubDomain];
      for (iDom=0; iDom<InfoProc->nbSubDomain; iDom++){
	Maillage->ListeDomaines[iDom].NbElements = 0;
	Maillage->ListeDomaines[iDom].Poids = 0;
	Maillage->ListeDomaines[iDom].NbVoisins = 0;
	Maillage->ListeDomaines[iDom].ListeVoisins = NULL;
	    }
    } /* end if Maillage->ListeDomaines == NULL */
      
    UnpackDom(TabTMP, TailleTab, InfoProc->Split_Comm, &Maillage->ListeDomaines[0]);
  }
  else { /* cas où ce n'est pas le maitre */
    EnvoieMessage(InfoProc, 0, TAG_MAILLAGEMAITRE, TabTMP, TailleTab);      
  }
  
  /* libération */
  free (TabTMP); 
  TabTMP = NULL;

  /* --------- */
  /* Réception */
  /* --------- */

  /* le maitre recoit toute les infos des autres noeuds (que lui même) */
  if (InfoProc->me == 0){
    Maillage->NbDomainesPleins = 0;       /* on va compter le nombre de domaines pleins */
    Maillage->NbProcsVides = 0;           /* on va compter le nombre de domaines vides */
    Maillage->Poids = 0.0;                // poids total
    Maillage->NbElements = 0;             // nombre d'éléments total dans le maillage

    /* on met les numéros les plus petits à la fin pour les utiliser en premier */
    for (iDom=InfoProc->nbSubDomain-1; iDom>=0; iDom--){
      if (iDom != 0){
	      
	TailleTab = 0; /* car on ne connait pas encore la taille */
	TabTMP = RecoitMessage(InfoProc, iDom, TAG_MAILLAGEMAITRE, &TailleTab);
	
	UnpackDom(TabTMP, TailleTab, InfoProc->Split_Comm, &Maillage->ListeDomaines[iDom]);
	      
	free ((void*) TabTMP); TabTMP = NULL;
      }

      /* on compte le nombre de domaines pleins et vides */
      if (Maillage->ListeDomaines[iDom].NbElements != 0)
	Maillage->NbDomainesPleins += 1;
      else
	Maillage->NbProcsVides += 1;

      Maillage->Poids      += Maillage->ListeDomaines[iDom].Poids;
      Maillage->NbElements += Maillage->ListeDomaines[iDom].NbElements;
    }

#ifdef ARCANE_DEBUG
    info()<<"   ***************************";
    info()<<"   * Apres MAJMaillageMaitre *";
#ifdef DEBUG
    AfficheMaillage (Maillage);
#endif
    info()<<"   ***************************";
    verifMaillageMaitre(Maillage);
#endif
  } /* end if me == maitre */
  
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
void SplitSDMeshPartitioner::
verifMaillageMaitre(StrucMaillage* Maillage)
{
#ifdef ARCANE_DEBUG
  info()<<"  on entre dans verifMaillageMaitre";
  
  StrucListeDomMail* ListeDomaines = Maillage->ListeDomaines;
  int NbDomaines = Maillage->NbDomainesPleins;

  for (int iDom=0; iDom<NbDomaines; iDom++){
    for (int j=0; j<ListeDomaines[iDom].NbVoisins; j++){ 
      
      int NbNoeudsInterface = ListeDomaines[iDom].ListeVoisins[j].NbNoeudsInterface;
      int iVois = ListeDomaines[iDom].ListeVoisins[j].NoDomVois;

      // recherche si le voisin existe et si le nombre de noeuds est identiques
      int k;
      for (k=0; k<ListeDomaines[iVois].NbVoisins && ListeDomaines[iVois].ListeVoisins[k].NoDomVois != iDom; k++)
      { }
      if (k==ListeDomaines[iVois].NbVoisins){
        printf("on ne trouve pas le numéro de voisin \n");
        printf("pour info: iDom = %d, iVois  = %d\n",iDom,iVois);
        perror() << "verifMaillageMaitre en erreur sur le voisinage !!!";
      }
      
      if (ListeDomaines[iVois].ListeVoisins[k].NbNoeudsInterface != NbNoeudsInterface){
        printf("on ne trouve pas le même nombre de noeuds entre les voisins \n");
        printf("pour info: iDom = %d, iVois  = %d, NbNoeudsInterface %d != %d\n"
               ,iDom,iVois,NbNoeudsInterface,ListeDomaines[iVois].ListeVoisins[k].NbNoeudsInterface);
        perror() << "verifMaillageMaitre en erreur sur le nombre de noeuds !!!";
      }
    }
  }

  info()<<"  on sort de verifMaillageMaitre";
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
  int iDomDep; /* indice du domaine trop plein qui sert de départ aux parcours frontaux */
  double PoidsMoyen;

  int tailleFP;
  int tailleFS;
  
  int* FrontPrec; // front de noeuds du graphe (domaines)
  int* FrontSuiv;
  int* FrontTMP;
  /* pour chacun des noeuds du graphe, donne le noeud précédent. 
     Cela va permettre depuis un noeud donné lors du parcours frontal de trouver un cheminement vers le noeuds de départ */
  int* ListeNoeudsPrec; 

  int* FiltreDomaine; /* pour marquer les domaines à marqueDomVu, marqueDomNonVu ou marqueDomVide */
  int marqueDomVu    =  1;
  int marqueDomNonVu =  0;
  int marqueDomVide  = -1; // normallement, il n'y e a pas

  double* PoidsSave;

#ifdef ARCANE_DEBUG
  info()<<" ------------------------------------";
  info()<<" on entre dans  MAJDeltaGlobal, tolerance = "<<tolerance;
  info()<<" ------------------------------------";
  info()<<" ......... Maillage Initial ...........";
  AfficheListeDomaines(ListeDomaines,NbDomaines);
#endif
  
  if (NbDomaines==0){
    info()<<" SplitSDMeshPartitioner::MAJDeltaGlobal : NbDomaines nul !";
    return;
  }
  if (Maillage->NbDomainesPleins<=1){
#ifdef ARCANE_DEBUG
    info()<<" \n = on sort de MAJDeltaGlobal sans rien faire (NbDomainesPleins = "<<Maillage->NbDomainesPleins;
    info()<<" ------------------------------------";
#endif
    return;
  }

  FiltreDomaine = (int*) malloc ((size_t) NbDomaines*sizeof(int));
  CHECK_IF_NOT_NULL(FiltreDomaine);

  /* on met à zero tous les Delta  par sécurité */
  for (i=0; i<NbDomaines; i++)
    for (int j=0; j<ListeDomaines[i].NbVoisins; j++)
      ListeDomaines[i].ListeVoisins[j].Delta = 0.0;
  
  /* 
     on décale le nombre de noeuds de chacun des domaines de manière à avoir 
     une somme totale <= 0 
  */

  for (i=0; i<NbDomaines; i++){
    if (ListeDomaines[i].NbElements != 0)
      FiltreDomaine[i] = marqueDomNonVu;
    else 
      FiltreDomaine[i] = marqueDomVide;
  }
  
  PoidsSave = (double*) malloc ((size_t) NbDomaines*sizeof(double));
  CHECK_IF_NOT_NULL(PoidsSave);

  /* la valeur moyenne */
  PoidsMoyen = Maillage->Poids/(double)Maillage->NbDomainesMax;

  /* on fait le décalage */  
  for (i=0; i<NbDomaines; i++){
    /* on met de côté le poids */
    PoidsSave[i] = ListeDomaines[i].Poids;
    
    if (FiltreDomaine[i] != marqueDomVide)
      ListeDomaines[i].Poids -= PoidsMoyen;
  } 

#ifdef DEBUG
  info()<<" Après Poids -= PoidsMoyen";
  AfficheListeDomaines(ListeDomaines,NbDomaines);
#endif

  /* initialisation mémoire au plus large */
  FrontPrec = (int*) malloc ((size_t) (NbDomaines-1)*sizeof(int));
  CHECK_IF_NOT_NULL(FrontPrec);
  FrontSuiv = (int*) malloc ((size_t) (NbDomaines-1)*sizeof(int));
  CHECK_IF_NOT_NULL(FrontSuiv);

  ListeNoeudsPrec = (int*) malloc ((size_t) (NbDomaines)*sizeof(int));
  CHECK_IF_NOT_NULL(ListeNoeudsPrec);

  for (iDomDep=0; iDomDep<NbDomaines; iDomDep++){
#ifdef DEBUG  
    info()<<" ListeDomaines[iDomDep = "<<iDomDep<<"].Poids  = "<<ListeDomaines[iDomDep].Poids;
#endif
	  
    /* on ne prend comme départ que les domaines étant au dessus de la moyenne en poids 
       (donc > 0 maintenant) */
    if (ListeDomaines[iDomDep].Poids > tolerance){
      /* DANS CE QUI SUIT, UN NOEUD EST UN DOMAINE, IL S'AGIT D'UN NOEUD DU GRAPHE */
      
      /* on initialise les domaines qui ont été vu précédemment à non vu */
      for (i=0; i<NbDomaines; i++)
	if (FiltreDomaine[i] == marqueDomVu)
	  FiltreDomaine[i] = marqueDomNonVu;
      
      /* on marque le noeud actuel comme étant vu */
      FiltreDomaine[iDomDep] = marqueDomVu;
      
      /* initialisation du front de départ */
      tailleFS = 1;
      FrontSuiv[tailleFS-1] = iDomDep;
      
#ifdef DEBUG 
      info()<<" FrontSuiv[0] = "<<iDomDep;
#endif
      
      /* initialisation du noeud de départ (pas de noeud précédent) */
      ListeNoeudsPrec[FrontSuiv[tailleFS-1]] = -1;
      
      /* boucle tant que le noeud de départ est trop gros (poids>0) */
      while (ListeDomaines[iDomDep].Poids > tolerance){
#ifdef DEBUG    
	info()<<" while (ListeDomaines["<<iDomDep<<"].Poids  = "<<ListeDomaines[iDomDep].Poids<<" > "<<tolerance;
#endif

	/* on permute les fronts suivant et précédents */
	FrontTMP = FrontPrec;
	FrontPrec = FrontSuiv;
	FrontSuiv = FrontTMP;
	tailleFP = tailleFS;
	tailleFS = 0; /* le front suivant est désormais vide */
	
	/* cas où le domaine de départ n'a pas réussi à dispercer son surplus sur
	   les autre domaines, et que l'on a vu tout ce que l'on pouvait */
	if (tailleFP == 0){
	  fatal()<<" partitionner/MAJDeltaGlobal: on ne trouve plus de domaine alors que l'on n'a pas terminé !!!";
	}
	
	/* 
	   on progresse d'un front 
	*/
	
	/* boucle sur les noeuds du front précédent */
	for (int iFP=0; iFP<tailleFP; iFP++){
	  int iDom = FrontPrec[iFP];
	  
	  /* boucle sur les voisins du noeud */
	  for (int iVois=0; iVois<ListeDomaines[iDom].NbVoisins; iVois++){
	    int iDomVois = ListeDomaines[iDom].ListeVoisins[iVois].NoDomVois;
	    if (FiltreDomaine[iDomVois] == marqueDomNonVu){
	      /* on marque ce noeud */
	      FiltreDomaine[iDomVois] = marqueDomVu;
	      ListeNoeudsPrec[iDomVois] = iDom;
#ifdef DEBUG    
	      info()<<"  FrontSuiv["<<tailleFS<<"] = "<<iDomVois;
#endif
	      FrontSuiv[tailleFS++] = iDomVois;
	    }
	  }
	} /* end for iFD<tailleFP */
	
	
	/* 
	   on décharge le noeud de base sur les noeuds du front suivant
	*/
	
	/* boucle sur les noeuds du front suivant */
	for (int iFS=0; iFS<tailleFS; iFS++){
	  int iDom = FrontSuiv[iFS];
	  
#ifdef DEBUG  
	  info()<<" ListeDomaines["<<iDom<<"].Poids = "<<ListeDomaines[iDom].Poids;
#endif
	  
	  /* on ne donne qu'aux noeuds déficitaires */
	  if (ListeDomaines[iDom].Poids < 0.0){
	    /* on ne donne pas plus que ce que l'on a */
	    double don = MIN(-ListeDomaines[iDom].Poids, ListeDomaines[iDomDep].Poids);
	    
	    /* cas où il y a un don */
	    if (don > 0.0){
	      int iDomTmp;
	      int iDomTmpPrec;
	      
	      ListeDomaines[iDom].Poids += don;
	      ListeDomaines[iDomDep].Poids -= don;
	      
	      //fprintf(stdout," don = %f\n",don);
	      
	      /* on remonte tout le chemin pour mettre à jour le Delta
		 entre les noeuds iDomTmp et iDomTmpPrec */
	      iDomTmp = iDom;
	      iDomTmpPrec = ListeNoeudsPrec[iDomTmp];
	      while (iDomTmpPrec != -1){
		/* on incrémente le Delta de don entre iDomTmpPrec et iDomTmp */
		MAJDelta(don,iDomTmpPrec,iDomTmp,ListeDomaines);
		/* de même (au signe pret) pour l'interface réciproque */
		MAJDelta(-don,iDomTmp,iDomTmpPrec,ListeDomaines);
		
		/* on remonte d'un cran en arrière sur le chemin */
		iDomTmp = iDomTmpPrec;
		iDomTmpPrec = ListeNoeudsPrec[iDomTmp];
		
	      } /* end while iDomTmpPrec != -1 */
	      
	    }/* end if don != 0 */
	  } /* end if NbNoeuds < 0, pour le domaine du front */
	} /* end for iFD<tailleFP */
	//	AfficheListeDomaines(ListeDomaines,NbDomaines);
	
      } /* end while Poids > tolerance */
    } /* end if Poids > tolerance */
  } /* end for iDomDep<NbDomaines */
  
 
  /* on remet les valeurs en place */  
  for (i=0; i<NbDomaines; i++)
    ListeDomaines[i].Poids = PoidsSave[i];
 
  free((void*) FrontPrec);       FrontPrec       = NULL;
  free((void*) FrontSuiv);       FrontSuiv       = NULL;
  free((void*) ListeNoeudsPrec); ListeNoeudsPrec = NULL;
  free((void*) FiltreDomaine);   FiltreDomaine   = NULL;
  free((void*) PoidsSave);       PoidsSave       = NULL;

#ifdef ARCANE_DEBUG
  info()<<" ......... Maillage Final ...........";
  AfficheListeDomaines(ListeDomaines,NbDomaines);
  info()<<" = on sort de   MAJDeltaGlobal     =";
  info()<<" ------------------------------------";
#endif
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
void SplitSDMeshPartitioner::
MAJDelta(double ajout, int iDom, int iVois, StrucListeDomMail* ListeDomaines)
{
  int j;
  for (j=0; 
       j<ListeDomaines[iDom].NbVoisins
	 &&
	 ListeDomaines[iDom].ListeVoisins[j].NoDomVois != iVois;
       j++)
    {}
  
  if (j == ListeDomaines[iDom].NbVoisins) 
    {
      info()<<"on ne trouve pas le numéro de voisin";
      info()<<"pour info: ajout = "<<ajout<<", iDom = "<<iDom<<", iVois = "<<iVois;
      pfatal()<<"Erreur dans Partitionner/MAJDelta, pas de voisin !";
    }

  /* on fait le décalage */
  ListeDomaines[iDom].ListeVoisins[j].Delta += ajout;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
double SplitSDMeshPartitioner::
CalculDeltaMin(StrucMaillage* Maillage, double deltaMin, int iterEquilibrage, int NbMaxIterEquil)
{
  // le but : limiter le deltaMin pour qu'il ne soit pas trop petit les premières fois
  // dans le cas où les transferts sont importants.
  // En effet, cela a tendance à morceler les domaines ce qui se répercute par de moins bonnes perf
  // (plus d'attentes lors des communications)

  debug()<<"  on entre dans SplitSDMeshPartitioner::CalculDeltaMin,  deltaMin = "<<deltaMin
         << ", iterEquilibrage = " <<iterEquilibrage
         << ", NbMaxIterEquil = " <<NbMaxIterEquil;
  if (arcaneIsDebug())
    AfficheMaillage (Maillage);

  double deltaAjuste = deltaMin;

  // recherche du plus grand Delta sur les interfaces
  double deltaMaxItf = 0.0;
  // idem pour la somme des deltas par domaine, ramené au poid du domaine (donc un ratio)
  double ratioDeltaMax = 0.0;

  StrucListeDomMail* ListeDomaines = Maillage->ListeDomaines;
  int NbDomaines = Maillage->NbDomainesPleins;

  for (int iDom=0; iDom<NbDomaines; iDom++){
    double deltaTotalDom = 0.0;
    for (int j=0; j<ListeDomaines[iDom].NbVoisins; j++){ 
      double delta = ListeDomaines[iDom].ListeVoisins[j].Delta;
      deltaMaxItf = MAX(delta, deltaMaxItf);
      if (delta > 0.0) deltaTotalDom += delta;
    }
    double ratio = 0.0;
    if (ListeDomaines[iDom].Poids>0.0) 
      ratio = deltaTotalDom/ ListeDomaines[iDom].Poids;
    ratioDeltaMax = MAX(ratio,ratioDeltaMax);
  }

  double poidsMoy =  Maillage->Poids/(double)Maillage->NbDomainesMax;

  // heuristique pour limiter deltaAjuste
  if (ratioDeltaMax>0.9)
    deltaAjuste = poidsMoy/3.0;
  else if (ratioDeltaMax>0.5)
    deltaAjuste = deltaMaxItf/10.0;

  // pour ne pas avoir plus petit que le min qui semble raisonnable et qui est paramétré par le cas .plt
  deltaAjuste = MAX(deltaMin,deltaAjuste);

#if defined(ARCANE_DEBUG) || defined(DEBUG_PARTITIONER)
  // que représente ce delta / poids moy ? tout ou une faible proportion ?
  double proportion = deltaMaxItf/poidsMoy;

  info()<<" deltaMaxItf = "<<deltaMaxItf;
  info()<<" ratioDeltaMax = "<<ratioDeltaMax;
  info()<<" poidsMoy = "<<poidsMoy;
  info()<<" proportion = "<<proportion;
  info()<<" deltaAjuste = "<<deltaAjuste;
#endif

  return deltaAjuste;
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
void SplitSDMeshPartitioner::
Equilibrage(StrucInfoProc* InfoProc, StructureBlocEtendu* Domaine, StrucMaillage* Maillage)
{
  int   iDom;             // indice pour les boucles
  int indDomCharge = -1;  // le domaine dont on va extraire des éléments
  int indDomVois = -1;    // le domaine qui va recevoir depuis indDomCharge
  
  double deltaMin;        // on ne fera le transfert que pour un poids minimum.
  // desequilibre-maximal (0.01) dans fichier .plt => maxImbalance()
  deltaMin = Maillage->Poids/(double)Maillage->NbDomainesMax*maxImbalance()/6.0; // on prend comme valeur maxImbalance/6 de la moyenne (donc 0.1 par défaut)

  double poidsMax = 0;    // pour la recherche du domaine le plus chargé

  void* TabTMP; /* pour transférer les infos sur les procs choisis et le Delta du maître vers les autres procs */

  int NbAppelsAEquil2Dom = -1;

  int iterEquilibrage = 0;
  int NbMaxIterEquil = 5; // phase d'équilibrage global. TODO voir s'il faut faire évoluer cette valeur (la paramétrer)

  double tolConnexite = 0.1; // un sous-domaine non connexe plus petit de tolConnexite*taille_moyenne, sera transféré


#ifdef ARCANE_DEBUG
  info()<<" -------------------------------------";
  info()<<"  on entre dans SplitSDMeshPartitioner::Equilibrage,  deltaMin = "<<deltaMin;
#endif

  int TailleTMP = TailleEquil();
  TabTMP = malloc((size_t)TailleTMP);
  CHECK_IF_NOT_NULL(TabTMP);

  /* on se limite à NbMaxIter itérations */		
  while (iterEquilibrage<NbMaxIterEquil && NbAppelsAEquil2Dom!=0){
    int* FiltreDomaine;
    int marqueDomVu    =  1;
    int marqueDomNonVu =  0;
    
    iterEquilibrage+=1;
    NbAppelsAEquil2Dom = 0;
    poidsMax = 0;
    indDomCharge = -1;
    

    int* MasqueDesNoeuds = GetMasqueDesNoeuds(InfoProc);
    int* MasqueDesElements = GetMasqueDesElements(InfoProc);
    int marqueVu = 0;
    int marqueNonVu = 0;

    IPrimaryMesh* mesh = this->mesh()->toPrimaryMesh();
    
    /* sur le maître */
    if (InfoProc->me == 0){
      info()<<" SplitSDMeshPartitioner::Equilibrage de la charge (iteration No "<<iterEquilibrage<<")";
      
      /* MAJ des Delta sur les interfaces de la description des domaines sur le maître */
      MAJDeltaGlobal(InfoProc, Maillage, deltaMin); // ex GetDeltaNoeuds

      // calcul un deltaMin en fonction des transferts locaux
      double deltaAjuste = CalculDeltaMin(Maillage, deltaMin, iterEquilibrage, NbMaxIterEquil);
      
      /* on va marquer les domaines vu (ou vide) */
      FiltreDomaine = (int*) calloc ((size_t) Maillage->NbDomainesMax, sizeof(int));
      CHECK_IF_NOT_NULL(FiltreDomaine);
      
      // pour les domaines vides
      for (iDom=0; iDom<Maillage->NbDomainesMax; iDom++)
	if (Maillage->ListeDomaines[iDom].NbElements == 0)
	  FiltreDomaine[iDom] = marqueDomVu;
      
      // boucle tant que l'on trouve un domaine sur lequel faire un transfert
      do {

	/* on recherche le plus chargé des domaines restant */
	indDomCharge = -1;
	poidsMax = 0;
	
	for (iDom=0; iDom<Maillage->NbDomainesMax; iDom++)
	  if (FiltreDomaine[iDom] == marqueDomNonVu && Maillage->ListeDomaines[iDom].Poids > poidsMax){
	    poidsMax = Maillage->ListeDomaines[iDom].Poids;
	    indDomCharge = iDom;
	  }

#ifdef ARCANE_DEBUG
	info()<<"indDomCharge = "<<indDomCharge<<"; poidsMax = "<<poidsMax;
#endif
	
	if (indDomCharge != -1) {
	  /* on ne veut plus le retrouver */
	  FiltreDomaine[indDomCharge] = marqueDomVu;
	  
	  /* on choisi 2 domaines pour effectuer le déplacement de l'interface entre les 2 */
	  
	  /* pour ce domaine chargé, on regarde vers quel autre domaine il pourrait 
	     y avoir transfert de noeuds */
	  for (int i=0; i<Maillage->ListeDomaines[indDomCharge].NbVoisins; i++){
	    indDomVois = Maillage->ListeDomaines[indDomCharge].ListeVoisins[i].NoDomVois;
	    
	    
	    /* on se limite aux transferts dont le DeltaNoeuds>deltaNoeudsMin */
	    if (Maillage->ListeDomaines[indDomCharge].ListeVoisins[i].Delta > deltaAjuste){
#if defined(ARCANE_DEBUG) || defined(DEBUG_PARTITIONER)
	      info()<<" Equilibrage ("<<iterEquilibrage<<") pour le couple indDomCharge = "<<indDomCharge<<"; indDomVois = "<<indDomVois
		    <<"; Delta = "<<Maillage->ListeDomaines[indDomCharge].ListeVoisins[i].Delta;
#endif
	      
	      /* on diffuse l'information à tous les processeurs */
	      PackEquil(InfoProc, indDomCharge, indDomVois, Maillage->ListeDomaines[indDomCharge].ListeVoisins[i].Delta,
			TabTMP, TailleTMP, InfoProc->Split_Comm);
	      
	      TabTMP = DiffuseMessage(InfoProc, 0, TabTMP, TailleTMP);
	      
	      // change la marque pour chaque appel (ce qui permet de libérer les noeuds)
	      marqueVu+=1;

	      // équilibrage entre 2 domaines
	      Equil2Dom(MasqueDesNoeuds, MasqueDesElements, marqueVu, marqueNonVu,
			InfoProc, Domaine, Maillage, indDomCharge, indDomVois, 
			Maillage->ListeDomaines[indDomCharge].ListeVoisins[i].Delta);
	      NbAppelsAEquil2Dom += 1;

	    } /* end if Delta > deltaAjuste */
	  } /* end for i<NbVoisins */
	  
	}// end if indDomCharge != -1
      } while (indDomCharge != -1);

      indDomVois = -1; /* pour informer que l'on arete, indDomCharge == -1 et indDomVois == -1 */
      double DeltaNul = 0.0;
      PackEquil(InfoProc, indDomCharge, indDomVois, DeltaNul, TabTMP, TailleTMP, InfoProc->Split_Comm);
      
      TabTMP = DiffuseMessage(InfoProc, 0, TabTMP, TailleTMP);

      free((void*) FiltreDomaine); 
      FiltreDomaine = NULL;


#ifdef ARCANE_DEBUG
      info()<<" NbAppelsAEquil2Dom = "<<NbAppelsAEquil2Dom;
#endif

    } /* end if me == 0 */
    else {
      double Delta;

      do {
	TabTMP = DiffuseMessage(InfoProc, 0, TabTMP, TailleTMP);      
	UnpackEquil(TabTMP, TailleTMP, InfoProc->Split_Comm, &indDomCharge, &indDomVois, &Delta);

	if (indDomCharge != -1){
	  // change la marque pour chaque appel
	  marqueVu+=1;

	  // équilibrage entre 2 domaines
	  Equil2Dom(MasqueDesNoeuds, MasqueDesElements, marqueVu, marqueNonVu,
		    InfoProc, Domaine, Maillage, indDomCharge, indDomVois, Delta);
	  NbAppelsAEquil2Dom += 1;
	}

      } while(indDomCharge != -1);

    } /* end else if me == 0 */
    
    // synchro dans le cas où il y a eu des modifs
    if (NbAppelsAEquil2Dom){
      // libération des noeuds dans les interfaces
      LibereDomaine(Domaine);
      
#ifdef ARCANE_DEBUG
      info() << "cells_new_owner.synchronize() et changeOwnersFromCells()";
#endif
      // on ne fait la synchro que pour une unique série d'appels à Equil2Dom
      VariableItemInt32& cells_new_owner = mesh->itemsNewOwner(IK_Cell);
      cells_new_owner.synchronize();
      changeOwnersFromCells();
      // effectue le transfert effectif des données d'un proc à l'autre, sans compactage des données
      bool compact = mesh->properties()->getBool("compact");
      mesh->properties()->setBool("compact", false);
      mesh->exchangeItems();
      mesh->properties()->setBool("compact", compact);
      
      // MAJ du Domaine
      MAJDomaine(Domaine);
      
      /* on remet à jour la structure Maillage->ListeDomaines */
      MAJMaillageMaitre(InfoProc,Domaine,Maillage);
    } // end if NbAppelsAEquil2Dom

    free((void*)MasqueDesNoeuds);   
    MasqueDesNoeuds   = NULL;
    free((void*)MasqueDesElements); 
    MasqueDesElements = NULL;	

    AfficheEquilMaillage(Maillage);

    // Rend les domaines connexes autant que possible suivant une tolérance 
    // (on accepte des partie non connexes à condition que leur taille soit > tol*taille_moyenne)
    marqueVu+=1;
    ConnexifieDomaine(InfoProc, Domaine, Maillage, tolConnexite);

    AfficheEquilMaillage(Maillage);
    
  } /* end while (iterEquilibrage<NbMaxIterEquil && NbAppelsAEquil2Dom!=0) */

  free(TabTMP); 
  TabTMP = nullptr;
  
  debug()<<" nombre d'iterations pour equilibrage  = "<<iterEquilibrage<<" / "<<NbMaxIterEquil<<" max ";
  debug()<<" = on sort de SplitSDMeshPartitioner::Equilibrage";
  debug()<<" -------------------------------------";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SplitSDMeshPartitioner::
Equil2Dom(int* MasqueDesNoeuds, int* MasqueDesElements, int marqueVu, int marqueNonVu,
          StrucInfoProc* InfoProc, StructureBlocEtendu* Domaine, StrucMaillage* Maillage,
          int indDomCharge, int indDomVois, double Delta)
{
  ARCANE_UNUSED(Maillage);

  debug()<<"    on entre dans Equil2Dom (indDomCharge:"<<indDomCharge<<", indDomVois:"<<indDomVois<<", Delta:"<<Delta;

  // tableau des éléments sélectionnés pour être déplacé
  Arcane::UniqueArray<Arcane::Cell> ListeElements;

  if (InfoProc->me == indDomCharge){
    int iIntf = 0;
    /* l'interface entre les deux domaines existe-t-elle  encore ? */
    for (iIntf=0; 
	 iIntf<Domaine->NbIntf
	   && Domaine->Intf[iIntf].NoDomVois != indDomVois; 
	 iIntf++)
      { }
  
    if (iIntf==Domaine->NbIntf){
#if defined(ARCANE_DEBUG) || defined(DEBUG_PARTITIONER)
      pinfo()<<"### l'interface a disparu ### entre "<<indDomCharge<< " et " <<indDomVois;
#endif	        
    }
    else {
      // sélection des éléments par un parcour frontal depuis l'interface entre les 2 sous-domaines.
      SelectElements(MasqueDesNoeuds, MasqueDesElements, 
		     marqueVu, marqueNonVu,
		     InfoProc, Domaine, Delta, indDomVois, ListeElements);

    }

    // marquage pour déplacement des données arcane
    IPrimaryMesh* mesh = this->mesh()->toPrimaryMesh();
    VariableItemInt32& cells_new_owner = mesh->itemsNewOwner(IK_Cell);
    
    for (int i=0; i<ListeElements.size(); i++){
      const Cell item = ListeElements[i];
      cells_new_owner[item] = indDomVois;
    }

  } // end if me == indDomCharge
#ifdef ARCANE_DEBUG
  else {
    info() << "SelectElements et autres operations sur les processeurs "<<indDomCharge<<" et " <<indDomVois;
  }

  info()<<"     on sort de    Equil2Dom";
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
  info()<<"SplitSDMeshPartitioner::SelectElements(Domaine, Delta = "<<Delta<<", indDomVois = "<<indDomVois<<")";
  info()<<" Domaine->NbElements = " << Domaine->NbElements;
  info()<<" Domaine->PoidsDom = " << Domaine->PoidsDom;
#endif

  if (Delta <= 0.0){
    perror() << "Delta <= 0 !!!";
  }

  IMesh* mesh = this->mesh();

  if (Delta>=Domaine->PoidsDom){
#ifdef ARCANE_DEBUG
    pinfo()<<" Tout le domaine est sélectionné sur le domaine "<<subDomain()->subDomainId()
	   <<", avec SelectElements, PoidsDom = " << Domaine->PoidsDom
	   <<", Delta = "<<Delta;
#endif
    // cas où tout le domaine est demandé
    // ce proceseur va se retrouver vide ! à moins qu'il n'y ait un transfert depuis un autre proc qui compense la perte
    CellGroup own_cells = mesh->ownCells();
     ENUMERATE_CELL(i_item,own_cells){
      const Cell item = *i_item;
      //MasqueDesElements[item.localId()] = marqueVu;
      ListeElements.add(item);
     }
  } else {
    // cas où il faut sélectionner un sous-ensemble
    
    int iIntf;      /* indice de l'interface pour le voisin indDomVois */
    int NbFrontsMax;
    int NbFronts;
    int* IndFrontsNoeuds;
    int* IndFrontsElements;

    for (iIntf=0; iIntf<Domaine->NbIntf && Domaine->Intf[iIntf].NoDomVois != indDomVois; iIntf++)
      { }
	  
    if (iIntf==Domaine->NbIntf) {
      pfatal()<<" SelectElements ne trouve pas l'interface parmis les voisins !!!";
    }

    NbFrontsMax = Domaine->NbElements / 2;
    
    /* les noeuds pris dans les fronts */
    Arcane::UniqueArray<Arcane::Node> FrontsNoeuds;

    /* les éléments pris dans les fronts */
    Arcane::Array<Arcane::Cell>& FrontsElements(ListeElements);

    IndFrontsNoeuds = (int*) malloc ((size_t) (NbFrontsMax+1)*sizeof(int));
    CHECK_IF_NOT_NULL(IndFrontsNoeuds);
    
    IndFrontsElements = (int*) malloc ((size_t) (NbFrontsMax+1)*sizeof(int));
    CHECK_IF_NOT_NULL(IndFrontsElements);

    /* front de départ pour le parcours */
    NbFronts = 1;

    /* on met comme front de départ la liste des noeuds dans l'interface */
    for (int i=0; i<Domaine->Intf[iIntf].ListeNoeuds.size(); i++)
      FrontsNoeuds.add(Domaine->Intf[iIntf].ListeNoeuds[i]);
    
    IndFrontsNoeuds[0] = 0;
    IndFrontsNoeuds[1] = Domaine->Intf[iIntf].ListeNoeuds.size();
    IndFrontsElements[0] = 0;
    IndFrontsElements[1] = 0;

    // on marque les mailles fantomes comme étant déjà vues
    int me = subDomain()->subDomainId();
    CellGroup all_cells = mesh->allCells(); // éléments sur ce processeur (mailles fantomes comprises)
    ENUMERATE_CELL(i_item,all_cells){
      const Cell cell = *i_item;
      if (cell.owner() != me)
	MasqueDesElements[cell.localId()] = marqueVu;
    }

    // on marque les éléments déjà sélectionné
    for (int i=0; i<ListeElements.size(); i++){
      const Cell cell = ListeElements[i];
      MasqueDesElements[cell.localId()] = marqueVu;
    }

    
    int retPF; 
    // Parcour Frontal pour un poids Delta demandé
    retPF = ParcoursFrontalDelta (MasqueDesNoeuds, MasqueDesElements, 
				  marqueVu, marqueNonVu,
				  Delta,
				  &NbFronts, NbFrontsMax,
				  FrontsNoeuds,  IndFrontsNoeuds,
				  FrontsElements,IndFrontsElements);

    /* le dernier front est un front incomplet, on le concatène à l'avant dernier pour le lissage */
    if (NbFronts>1){
      IndFrontsNoeuds[NbFronts-1] = IndFrontsNoeuds[NbFronts];
      IndFrontsElements[NbFronts-1] = IndFrontsElements[NbFronts];
      NbFronts-=1;
    }
    
    /* on fait le lissage du front dans le cas où l'on n'a pas sélectionné tout le domaine 
       et que le ParcoursFrontal s'est bien déroulé (n'est pas bloqué) */
    if (FrontsNoeuds.size() < Domaine->NbElements && retPF == 0)
      LissageDuFront(MasqueDesNoeuds, MasqueDesElements, 
		     marqueVu, marqueNonVu,
		     NbFronts,
		     FrontsNoeuds,  IndFrontsNoeuds,
		     FrontsElements,IndFrontsElements);


    free((void*)IndFrontsNoeuds);   IndFrontsNoeuds   = NULL;
    free((void*)IndFrontsElements); IndFrontsElements = NULL;
  }

  info()<<" en sortie: ListeElements.size() = "<<ListeElements.size();
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
int SplitSDMeshPartitioner::
ParcoursFrontalDelta (int* MasqueDesNoeuds, int* MasqueDesElements, 
		      int marqueVu, int marqueNonVu,
		      double Delta,
		      int *pNbFronts, int NbFrontsMax,
		      Arcane::Array<Arcane::Node>& FrontsNoeuds,  int* IndFrontsNoeuds,
		      Arcane::Array<Arcane::Cell>& FrontsElements,int* IndFrontsElements)
{
#ifdef ARCANE_DEBUG
  info()<<"       = on entre dans ParcoursFront  :   (NbFronts = "<<*pNbFronts<<", NbFrontsMax = "<<NbFrontsMax<<")";
  info()<<"  FrontsNoeuds.size() = "<<FrontsNoeuds.size();
  info()<<"  FrontsElements.size() = "<<FrontsElements.size();
#endif

  int IndFn = 0; /* indices sur les tableaux [Ind]FrontsNoeuds */
  int IndFe = 0; /* indices sur les tableaux [Ind]FrontsElements */
  double PoidsActuel = 0.0;
  bool  bloque = false;

  /* on marque les noeuds et les éléments déjà dans les fronts (pour ne pas les avoir deux fois) */
  for (IndFn=0; IndFn<IndFrontsNoeuds[*pNbFronts]; IndFn++){
    MasqueDesNoeuds[FrontsNoeuds[IndFn].localId()] = marqueVu;
  }
  
  for (IndFe=0; IndFe<IndFrontsElements[*pNbFronts]; IndFe++){
    MasqueDesElements[FrontsElements[IndFe].localId()] = marqueVu;
  }

  /* on met dans le front initial les noeuds liés aux éléments de ce même front 
     si cela n'a pas été fait */
  if (IndFrontsElements[*pNbFronts] > 0 && IndFrontsNoeuds[*pNbFronts] == 0){
    //info()<<" Initialisation à partir de "<<IndFrontsElements[*pNbFronts]<<" éléments";
    for (int ielm=0; ielm<IndFrontsElements[*pNbFronts]; ielm++){
      const Cell cell = FrontsElements[ielm];
      PoidsActuel += m_poids_aux_mailles[cell];

      for (int iepn = 0; iepn < cell.nbNode(); iepn++){
	const Node nodeVois = cell.node(iepn); // noeud voisin  // int NoeudVoisin
	
	/* pour chaques nouveau noeud, on l'insère dans le nouveau front*/
	if (MasqueDesNoeuds[nodeVois.localId()] == marqueNonVu){
	  FrontsNoeuds.add(nodeVois);
	  IndFn+=1;
	  MasqueDesNoeuds[nodeVois.localId()] = marqueVu;
	}
      } /* end for iepn */
    }
    /* on met les noeuds dans le dernier front existant */
    IndFrontsNoeuds[*pNbFronts] = IndFn;
  }

  /*----------------------------------------------------------------*/
  /* boucle jusqu'à ce que l'on ait vu assez de noeuds ou de fronts */ 
  /*----------------------------------------------------------------*/
  do {
    /* pour chaques noeuds du front précédent, on regarde les éléments qu'il possède */
    for (int in = IndFrontsNoeuds[*pNbFronts-1];
	 in < IndFrontsNoeuds[*pNbFronts] && PoidsActuel<Delta;
	 in++) {
      const Node node = FrontsNoeuds[in]; // noeud du front précédent  // int Noeud
      
      /* on évite de s'arrêter sans avoir tous les éléments liés, 
	 on risquerait d'avoir un élément relié par un seul noeud ! */
      for (int inpe = 0; inpe < node.nbCell(); inpe++){
	const Cell cell = node.cell(inpe); //élément lié à ce noeud  //int Element
	
	/* pour ce nouvel élément, on l'insère dans le nouveau front 
	   et on regarde les noeuds qu'il possède */
	if (MasqueDesElements[cell.localId()] == marqueNonVu) {
	  FrontsElements.add(cell); 
	  IndFe+=1;
	  MasqueDesElements[cell.localId()] = marqueVu;
	
	  PoidsActuel += m_poids_aux_mailles[cell];
	  
	  for (int iepn = 0; iepn < cell.nbNode(); iepn++){
	    const Node nodeVois = cell.node(iepn); // noeud voisin  // int NoeudVoisin
		    
	    /* pour chaques nouveau noeud, on l'insère dans le nouveau front*/
	    if (MasqueDesNoeuds[nodeVois.localId()] == marqueNonVu){
	      FrontsNoeuds.add(nodeVois);
	      IndFn+=1;
	      MasqueDesNoeuds[nodeVois.localId()] = marqueVu;
	    } /* end if MasqueDesNoeuds == marqueNonVu */
	  } /* end for iepn */
	} /* end if MasqueDesElements == marqueNonVu */
      } /* end for inpe */
    } /* end for in */
    
    /* test du cas où cela se bloquerait */
    if (IndFrontsNoeuds[*pNbFronts-1] == IndFrontsNoeuds[*pNbFronts]){
      bloque = true;
    }

    *pNbFronts += 1;
    IndFrontsNoeuds[*pNbFronts] = IndFn;
    IndFrontsElements[*pNbFronts] = IndFe;

  }  
  while (*pNbFronts<NbFrontsMax && PoidsActuel<Delta && !bloque);


#ifdef ARCANE_DEBUG
  info()<<" NbFronts = "<<*pNbFronts;
  info()<<" Delta = "<<Delta;
  info()<<" PoidsActuel = "<<PoidsActuel;
  info()<<" NbNoeuds   = "<<IndFrontsNoeuds[*pNbFronts];
  info()<<" NbElements = "<<IndFrontsElements[*pNbFronts];
  info()<<" bloque = "<<(bloque?"VRAI":"FAUX");

  if (!(*pNbFronts<NbFrontsMax)){
    info()<<"       =  on arete apres avoir obtenu le nombre maximum de fronts "<<NbFrontsMax;
  }
  else if (!(PoidsActuel<Delta)){
    info()<<"       =  on arete apres avoir obtenu le poids desire "<<PoidsActuel;
  }
  else if (bloque){
    info()<<"       =  on est bloque (non connexe ?) =";
  }
  else{
    info()<<"       =  on arete parce que l'on a tout vu =";
  }
  info()<<"       = on sort de    ParcoursFront  .   =";
#endif

  return ((bloque)?1:0);
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SplitSDMeshPartitioner::
LissageDuFront (int* MasqueDesNoeuds, int* MasqueDesElements, 
                int marqueVu, int marqueNonVu,
                int NbFronts,
                Arcane::Array<Arcane::Node>& FrontsNoeuds,  int* IndFrontsNoeuds,
                Arcane::Array<Arcane::Cell>& FrontsElements,int* IndFrontsElements)
{
  ARCANE_UNUSED(IndFrontsElements);

  debug()<<"       on entre dans LissageDuFront  :    NbFronts = "<<NbFronts;

  int NbElementsAjoutes = 0;
  
  Arcane::UniqueArray<Arcane::Cell> ElementsALiberer;

  /* Récupération des éléments pris dans le dernier front de noeuds */
  for (int IndFn=IndFrontsNoeuds[NbFronts-1]; IndFn<IndFrontsNoeuds[NbFronts]; IndFn++){
    const Node node = FrontsNoeuds[IndFn]; // noeud du front précédent  // int Noeud
    
    for (int inpe = 0; inpe < node.nbCell(); inpe++){
      const Cell cell = node.cell(inpe); //élément lié à ce noeud  //int Element
	    
      /* si cet élément est non-vu */
      if (MasqueDesElements[cell.localId()] == marqueNonVu){
	/* on cherche si tous les noeuds de cet élément sont marqués vu */
	int iepn;
	for (iepn = 0; iepn < cell.nbNode() && MasqueDesNoeuds[cell.node(iepn).localId()] == marqueVu; iepn++)
	  { }
	  /* cas où tous les noeuds sont marqués */
	  if (iepn == cell.nbNode()){
	    /* on ajoute l'élément au dernier front */

	    FrontsElements.add(cell); 
	    NbElementsAjoutes+=1;
	    MasqueDesElements[cell.localId()] = marqueVu;
	  }
	  else {
	    /* on marque l'élement */
	    MasqueDesElements[cell.localId()] = marqueVu; // pour ne pas le reprendre tout de suite
	    ElementsALiberer.add(cell);
	  }
	      
	} /* end if Element non vu */
      } /* end for inpe ... */
    } /* end for IndFn ... */
  
  // on remet à marqueNonVu ceux que l'on a marqué juste pour ne les voir qu'une fois
  for (int i=0; i<ElementsALiberer.size(); i++)
    MasqueDesElements[ElementsALiberer[i].localId()] = marqueNonVu;

#ifdef ARCANE_DEBUG
  info()<<"       on sort de  LissageDuFront ("<<NbElementsAjoutes<<" elements ajoutes)";
#endif
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
void SplitSDMeshPartitioner::
ConnexifieDomaine(StrucInfoProc* InfoProc, StructureBlocEtendu* Domaine, StrucMaillage* Maillage, 
		  double tolConnexite)
{
#ifdef ARCANE_DEBUG
  info()<<"    on entre dans ConnexifieDomaine, tolConnexite = "<<tolConnexite;
#endif
  
  int* MasqueDesNoeuds = GetMasqueDesNoeuds(InfoProc);
  int* MasqueDesElements = GetMasqueDesElements(InfoProc);
  int marqueVu = 1;
  int marqueNonVu = 0;

  // Maillage->NbElements n'est connu que sur le maitre !
//   double tailleMoy = (double)Maillage->NbElements / (double)Maillage->NbDomainesMax;
  double tailleMoy = (double)Domaine->NbElements;

  // on marque les mailles fantomes comme étant déjà vues
  int me = InfoProc->me;
  IPrimaryMesh* mesh = this->mesh()->toPrimaryMesh();
  CellGroup all_cells = mesh->allCells(); // éléments sur ce processeur (mailles fantomes comprises)

  ENUMERATE_CELL(i_item,all_cells){
    Cell cell = *i_item;
    if (cell.owner() != me)
      MasqueDesElements[cell.localId()] = marqueVu;
  } // end ENUMERATE_CELL

  int NbElementsVus = 0;
  int NbElementsAVoir = Domaine->NbElements;
#ifdef ARCANE_DEBUG
  info()<<" NbElementsAVoir = "<<NbElementsAVoir;
#endif
  
  int NbFrontsMax;
  int NbFronts;
  int* IndFrontsNoeuds;
  int* IndFrontsElements;
  
  NbFrontsMax = Domaine->NbElements / 2;
  
  /* les noeuds pris dans les fronts */
  Arcane::UniqueArray<Arcane::Node> FrontsNoeuds;
  
  /* les éléments pris dans les fronts */
  Arcane::SharedArray<Arcane::SharedArray<Arcane::Cell> > ListeFrontsElements;
  
  IndFrontsNoeuds = (int*) malloc ((size_t) (NbFrontsMax+1)*sizeof(int));
  CHECK_IF_NOT_NULL(IndFrontsNoeuds);
  
  IndFrontsElements = (int*) malloc ((size_t) (NbFrontsMax+1)*sizeof(int));
  CHECK_IF_NOT_NULL(IndFrontsElements);


  /* on boucle de manière à voir tous les éléments du domaine */
  while (NbElementsVus < NbElementsAVoir){

    FrontsNoeuds.clear(); 
    Arcane::SharedArray<Arcane::Cell> FrontsElements;

    /* recherche d'un éléments non vu (il est mis dans le premier front) */
    //TODO à optimiser ?
    bool trouve = false;
    ENUMERATE_CELL(i_item,all_cells){
      const Cell cell = *i_item;
      if (!trouve && MasqueDesElements[cell.localId()] == marqueNonVu){
	FrontsElements.add(cell);
	trouve = true;
      }
    }
    if (!trouve){
      pfatal()<<"ConnexifieDomaine est bloqué lors de la recherche d'un élément de départ";
    }

    NbFronts = 1;
    IndFrontsNoeuds[0] = 0;
    IndFrontsNoeuds[1] = 0;
    IndFrontsElements[0] = 0;
    IndFrontsElements[1] = 1;

    /* recherche de l'ensemble connexe d'éléments non vus associé */
    ParcoursFrontalDelta (MasqueDesNoeuds, MasqueDesElements, 
			  marqueVu, marqueNonVu,
			  Domaine->PoidsDom,
			  &NbFronts, NbFrontsMax,
			  FrontsNoeuds,  IndFrontsNoeuds,
			  FrontsElements,IndFrontsElements);

    /* on met de côté cet ensemble d'éléments */
    ListeFrontsElements.add(FrontsElements);

    NbElementsVus+=FrontsElements.size();
#ifdef ARCANE_DEBUG
    info()<<"  NbElementsVus+="<<FrontsElements.size();
#endif

  } /* end while (NbElementsVus < NbElementsAVoir) */

  // nombre de composantes transférées
  int nbCCTransf = 0;

  /* s'il y a plus d'une composante connexe */
  if (ListeFrontsElements.size() > 1){
 
#ifdef ARCANE_DEBUG
    info()<<"    NbComposantesConnexes = "<<ListeFrontsElements.size();
#endif

    // analyse du nombre de domaines en dessous du seuil
    int nbDomEnDessous = 0;
    int plusGrosseCC = 0; // plus grosse composante connexe

    int seuil = (int)(tailleMoy*tolConnexite);
#ifdef ARCANE_DEBUG
    info()<< "  seuil = "<<seuil;
#endif

    for (int i=0; i<ListeFrontsElements.size(); i++){
      Arcane::Array<Arcane::Cell> &FrontsElements = ListeFrontsElements[i];
      plusGrosseCC = MAX(plusGrosseCC,FrontsElements.size());
      if (FrontsElements.size()<seuil)
	nbDomEnDessous+=1;
    }
    
#ifdef ARCANE_DEBUG
    info()<< "  nbDomEnDessous = "<<nbDomEnDessous;
#endif

    // pour éviter de prendre toutes les composantes
    if (nbDomEnDessous == ListeFrontsElements.size()){
#ifdef ARCANE_DEBUG
      info() <<" seuil abaissé à "<< plusGrosseCC;
#endif
      seuil = plusGrosseCC;
    }
    
    // pour chacune des composante de taille < seuil, on fait le transfert vers un voisin
    VariableItemInt32& cells_new_owner = mesh->itemsNewOwner(IK_Cell);
    for (int i=0; i<ListeFrontsElements.size(); i++){
      Arcane::Array<Arcane::Cell> &FrontsElements = ListeFrontsElements[i];
      if (FrontsElements.size()<seuil){
	nbCCTransf += 1;
	
	// recherche du sous-domaine voisin ayant un max de face en commun
	int indDomVois = getDomVoisMaxFace(FrontsElements, me);
	
	// on affecte les cellules à ce voisin
	for (int j=0; j<FrontsElements.size(); j++){
	  const Cell item = FrontsElements[j];
	  cells_new_owner[item] = indDomVois;
	}

      } // if FrontsElements.size()<seuil
    } // end for i<ListeFrontsElements.size()
#ifdef ARCANE_DEBUG
    info() << " Nombre de composantes transférées : "<<nbCCTransf;
#endif

  } // end if ListeFrontsElements.size() > 1

  /* libérations mémoire */
  free((void*) IndFrontsNoeuds);
  IndFrontsNoeuds = NULL;
  free((void*) IndFrontsElements);
  IndFrontsElements = NULL;
  
  free((void*)MasqueDesNoeuds);   
  MasqueDesNoeuds   = NULL;
  free((void*)MasqueDesElements); 
  MasqueDesElements = NULL;	


  // on ne fait la synchro que si l'un des sous-domaines à fait une modif
  // faire la somme sur tous les proc de nbCCTransf

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
  info()<<" ConnexifieDomaine: nbCCTransfSum = "<<nbCCTransfSum;
#endif

  if (synchroNecessaire){
#ifdef ARCANE_DEBUG
    info()<<"    on fait la synchronisation";
#endif
    // on ne fait la synchro que pour une unique série d'appels à Equil2Dom
    VariableItemInt32& cells_new_owner = mesh->itemsNewOwner(IK_Cell);
    cells_new_owner.synchronize();
    changeOwnersFromCells();
    // effectue le transfert effectif des données d'un proc à l'autre, sans compactage des données
    bool compact = mesh->properties()->getBool("compact");
    mesh->properties()->setBool("compact", false);
    mesh->exchangeItems();
    mesh->properties()->setBool("compact", compact);
    
    // MAJ du Domaine
    MAJDomaine(Domaine);
    
    /* on remet à jour la structure Maillage->ListeDomaines */
    MAJMaillageMaitre(InfoProc,Domaine,Maillage);
  }
#ifdef ARCANE_DEBUG
  info()<<"    on sort de ConnexifieDomaine";
#endif
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
int SplitSDMeshPartitioner::
getDomVoisMaxFace(Arcane::Array<Arcane::Cell>& ListeElements, int me)
{
  int indDomVois = -1;

  // nombre de face par sous-domaine voisin
  std::map<int,int> indVois_nbFace;

  for (int i=0; i<ListeElements.size(); i++){
    const Cell cell = ListeElements[i];

    for (int j=0;j<cell.nbFace(); j++){
      const Face face = cell.face(j);

      if (!face.isSubDomainBoundary()){
	int id1 = face.backCell().owner();
	int id2 = face.frontCell().owner();
	
	if (id1 == me && id2 != me)
	  indVois_nbFace[id2]+=1;
	else if (id1 != me && id2 == me)
	  indVois_nbFace[id1]+=1;
      }
    }
  }

  // recherche du plus grand nombre de faces
  int maxNbFaces = 0;
  std::map<int,int>::iterator iter;
  for (iter = indVois_nbFace.begin();
       iter != indVois_nbFace.end();
       ++iter){
    int nbFaces = (*iter).second;
    if (nbFaces>maxNbFaces){
      maxNbFaces = nbFaces;
      indDomVois = (*iter).first;
    }
  }
  
  if (indDomVois == -1)
    pfatal()<<"indDomVois toujours à -1 !!!";

#ifdef ARCANE_DEBUG
  pinfo()<<" getDomVoisMaxFace, me = "<<me<<", ListeElements.size() = "<<ListeElements.size()
	 <<", indDomVois = "<<indDomVois<<", maxNbFaces = "<<maxNbFaces;
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
  // recherche des plus grand id
  int maxNodeLocalId = 0;

  NodeGroup all_nodes = mesh->allNodes();
  ENUMERATE_NODE(i_item,all_nodes){
    const Node node = *i_item;
    maxNodeLocalId = MAX(maxNodeLocalId,node.localId());
  }
#ifdef ARCANE_DEBUG
  info() << "SplitSDMeshPartitioner::GetMasqueDesNoeuds(), maxNodeLocalId = "<<maxNodeLocalId;
#endif

  /* allocation et initialisation à 0 du masque */
  MasqueDesNoeuds = (int*) calloc ((size_t)(maxNodeLocalId+1), sizeof(int));
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
  // recherche des plus grand id
  int maxCellLocalId = 0;
  
  CellGroup all_cells = mesh->allCells(); // éléments sur ce processeur (mailles fantomes comprises)
  ENUMERATE_CELL(i_item,all_cells){
    const Cell cell = *i_item;
    maxCellLocalId = MAX(maxCellLocalId,cell.localId());
  }
  
#ifdef ARCANE_DEBUG
  info() << "SplitSDMeshPartitioner::GetMasqueDesElements(), maxCellLocalId = "<<maxCellLocalId;
#endif
  
  /* allocation et initialisation à 0 du masque */
  MasqueDesElements = (int*) calloc ((size_t)(maxCellLocalId+1), sizeof(int));
  CHECK_IF_NOT_NULL(MasqueDesElements);

  return MasqueDesElements;
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SplitSDMeshPartitioner::
LibereInfoProc(StrucInfoProc* &InfoProc)
{
  ARCANE_UNUSED(InfoProc);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SplitSDMeshPartitioner::
LibereDomaine(StructureBlocEtendu* &Domaine)
{
#ifdef ARCANE_DEBUG
  info()<<"LibereDomaine(...)";
#endif

  if (Domaine->NbIntf != 0){
    if (Domaine->Intf != NULL){
      for (int i=0; i<Domaine->NbIntf; i++)
	Domaine->Intf[i].ListeNoeuds.clear();
      delete [] Domaine->Intf;
      Domaine->Intf = NULL;
    }
  }

  Domaine->NbIntf     = 0;
  Domaine->NbElements = 0;
  Domaine->PoidsDom   = 0.0;
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
void SplitSDMeshPartitioner::
LibereMaillage(StrucMaillage* &Maillage)
{
#ifdef ARCANE_DEBUG
  info()<<"LibereMaillage(...)";
#endif

  if (Maillage->NbElements != 0){
    if (Maillage->ListeDomaines != NULL){
      for (int i=0; i<Maillage->NbDomainesMax; i++){
	if (Maillage->ListeDomaines[i].ListeVoisins != NULL){
	  delete [] Maillage->ListeDomaines[i].ListeVoisins;
	  Maillage->ListeDomaines[i].ListeVoisins = NULL;
	}
      }
      delete [] Maillage->ListeDomaines; 
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
AfficheDomaine (int NbDom, StructureBlocEtendu* Domaine)
{
  int i;
  int idom;

  for (idom=0; idom<NbDom; idom++) {
    info()<<" --------------------";
    info()<<" --- Domaine ("<<idom<<") ---";
    info()<<" --------------------";
    
    if (Domaine == NULL){
      info()<<" Domaine vide !  (pointeur NULL)";
    }
    else if (Domaine[idom].NbElements == 0) {
      info()<<" Domaine vide !  (pas d'éléments)";
    } 
    else {
      info()<<" NbElements = "<<Domaine[idom].NbElements;
      info()<<" PoidsDom   = "<<Domaine[idom].PoidsDom;
      info()<<" Interfaces (NbIntf = "<<Domaine[idom].NbIntf<<") :";
      for (i=0; i<Domaine[idom].NbIntf; i++) {
	info()<<" ("<<i<<") NoDomVois = "<<Domaine[idom].Intf[i].NoDomVois
		 <<", nb de noeuds "<<Domaine[idom].Intf[i].ListeNoeuds.size();
      } 
	  
    } /* end else if 'domaine vide' */
  } /* end for idom */
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
void SplitSDMeshPartitioner::
AfficheMaillage (StrucMaillage* Maillage)
{
  info()<<" ----------------";
  info()<<" --- Maillage ---";
  info()<<" ----------------";
  if (Maillage==NULL){
    info()<<" structure Maillage vide !";
  }
  else {
    info()<<" NbElements (total)   = "<<Maillage->NbElements;
    info()<<" Poids      (total)   = "<<Maillage->Poids;
    info()<<" NbDomainesMax        = "<<Maillage->NbDomainesMax;
    info()<<" NbDomainesPleins     = "<<Maillage->NbDomainesPleins;
    info()<<" NbProcsVides         = "<<Maillage->NbProcsVides;
    
    if (Maillage->ListeDomaines == NULL) {
      info()<<" Maillage.ListeDomaines == NULL";
    }
    else {
      AfficheListeDomaines(Maillage->ListeDomaines,Maillage->NbDomainesMax);
    }

  } /* else if Maillage==NULL */
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
void SplitSDMeshPartitioner::
AfficheListeDomaines(StrucListeDomMail* ListeDomaines, int NbDomaines)
{
  info()<<"  ListeDomaines :";
  for (int i=0; i<NbDomaines; i++) {
    info()<<" ("<<i<<") NbElements  = "<<ListeDomaines[i].NbElements<<"; Poids    = "<<ListeDomaines[i].Poids;
    info()<<" ("<<i<<") NbVoisins = "<<ListeDomaines[i].NbVoisins<<"; ListeVoisins :";
      for (int j=0; j<ListeDomaines[i].NbVoisins; j++) {
	info()<<" ("<<i<<")   NoDomVois  = "<<ListeDomaines[i].ListeVoisins[j].NoDomVois
		 <<"; NbNoeudsInterface  = "<<ListeDomaines[i].ListeVoisins[j].NbNoeudsInterface
		 <<"; Delta  = "<<ListeDomaines[i].ListeVoisins[j].Delta;
      } 
  }
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
void SplitSDMeshPartitioner::
AfficheEquilMaillage(StrucMaillage* Maillage)
{
  info()<<" AfficheEquilMaillage(...)";

  double poidsMin = 0.0;
  double poidsMax = 0.0;  

  if(Maillage->ListeDomaines != NULL){
    
    poidsMin = Maillage->ListeDomaines[0].Poids;
    
    for (int i=0; i<Maillage->NbDomainesMax; i++){
      double poidsDom = Maillage->ListeDomaines[i].Poids;

      if (poidsDom > poidsMax)
	poidsMax = poidsDom;
      if (poidsDom < poidsMin)
	poidsMin = poidsDom;
    }

    info()<<"   INFO equilibrage / noeuds : max : "<<poidsMax<<", min : "<<poidsMin
	     <<", max/moy = "<<poidsMax/(Maillage->Poids/(double)Maillage->NbDomainesMax);
  }
  else{
    info()<<"AfficheEquilMaillage : Maillage->ListeDomaines == NULL";
  }
  
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/** 
  Reçoit un tableau de taille (in)connue via MPI.
  Alloue ce tableau et le retourne (en sortie).

  {\em Remarque:} la taille du tableau est positive lorsqu'elle est connue sinon il est fait appel 
  aux fonctions MPI\_Probe et MPI\_Get\_count pour connaître la taille.

  @memo    Réception d'un tableau à l'aide de la libriarie de communication MPI.
  @param   InfoProc (I) structure décrivant le processeur sur lequel tourne l'application.
  @param   FromProc (I) numéro du processeur qui envoit.
  @param   Tag      (I) marque pour distinguer les messages.
  @param   pTailleTMP (I/O) pointeur sur taille en octets du tableau TabTMP:\\
               <=0 si la taille du tableau est inconnue\\
	       > 0 si la taille du tableau est connue.

  @return  (void*) TabTMP : tableau qui est envoyé.
  @see     MAJMaillageMaitre
  @author  Eric Brière de l'Isle, ONERA, DRIS/SRL
  @version Création Septembre 1996 - 26/08/98
*/
void* SplitSDMeshPartitioner::RecoitMessage (StrucInfoProc* InfoProc, int FromProc, int Tag, int *pTailleTMP)
{
  void *TabTMP;
  int ierror;        /* retour d'erreur sur lib MPI */
  MPI_Status status; /* pour les communication MPI  */


  if (*pTailleTMP <= 0)
    {
      /* pour récupérer l'info sur la taille des tableaux qui vont suivre,
	 on attend qu'un message soit arrivé */
      ierror = MPI_Probe (FromProc,
			  Tag,
			  InfoProc->Split_Comm,
			  &status);
      
      if (ierror != MPI_SUCCESS) 
	{
	  InfoProc->m_service->pfatal()<<"Problème sur "<<InfoProc->me<<" de communication en provenance de "
				       <<FromProc<<", lors de MPI_Probe";
	}
      
      /* récupération de la taille du message | tableau */
      ierror = MPI_Get_count (&status,
			      MPI_PACKED,
			      pTailleTMP);
      
      if (ierror != MPI_SUCCESS) {
	InfoProc->m_service->pfatal()<<"Problème sur "<<InfoProc->me<<" de communication en provenance de "
				     <<FromProc<<", lors de MPI_Get_count";
      }
    } /* end if *pTailleTMP <= 0 */

  /* allocation pour le tableau tampon */
  if (*pTailleTMP > 0)
    {
      TabTMP = malloc ((size_t) *pTailleTMP);
      CHECK_IF_NOT_NULL(TabTMP);
    }
  else
    TabTMP = NULL;

  /* réception du message */
  ierror = MPI_Recv (TabTMP,
		     *pTailleTMP,
		     MPI_PACKED,
		     FromProc,
		     Tag,
		     InfoProc->Split_Comm,
		     &status);
  
  if (ierror != MPI_SUCCESS) {
      InfoProc->m_service->pfatal()<<"Problème sur "<<InfoProc->me<<" de communication en provenance de "
				   <<FromProc<<", lors de MPI_Recv";
    }

  return (TabTMP);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/** 
  Fonction qui se charge d'envoyer un tableau.

  {\em Remarque:} La taille du tableau est en octets.

  @memo    Envoi d'un message.
  @param   InfoProc  (I) structure décrivant le processeur sur lequel tourne l'application.
  @param   ToProc    (I) noeud de calcul de destination.
  @param   Tag       (I) marque pour différencier les messages.
  @param   TabTMP    (I) tableau qui est envoyé.
  @param   TailleTMP (I) taille de TabTMP (en octets).
  @return  void
  @see     Equilibrage,MAJMaillageMaitre
  @author  Eric Brière de l'Isle, ONERA, DRIS/SRL
  @version Création Janvier 1997 - 26/08/98
*/
void SplitSDMeshPartitioner::EnvoieMessage(StrucInfoProc* InfoProc, int ToProc, int Tag, void* TabTMP, int TailleTMP)
{
  int ierror; /* retour d'erreur sur les fonctions MPI */
  
  ierror = MPI_Send ((void *) TabTMP,
		     TailleTMP,
		     MPI_PACKED,
		     ToProc,
		     Tag,
		     InfoProc->Split_Comm);
  
  if (ierror != MPI_SUCCESS) {
    InfoProc->m_service->pfatal()<<"Problème sur "<<InfoProc->me<<" de communication vers "
				 <<ToProc<<", lors de MPI_Send";
  }
  

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// /** 
//   Fonction qui se charge d'envoyer un tableau avec une communication non bloquante.

//   {\em Remarque:} La taille du tableau est en octets.

//   @memo    Envoi d'un message non-bloquants.
//   @param   InfoProc  (I) structure décrivant le processeur sur lequel tourne l'application.
//   @param   ToProc    (I) noeud de calcul de destination.
//   @param   Tag       (I) marque pour différencier les messages.
//   @param   TabTMP    (I) tableau qui est envoyé.
//   @param   TailleTMP (I) taille de TabTMP (en octets).
//   @return  (MPI\_Request*) prequest : pointeur sur une structure propre à MPI pour la gestion du message.
//   @see     
//   @author  Eric Brière de l'Isle, ONERA, DRIS/SRL
//   @version Création Décembre 1998
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
//     InfoProc->m_service->pfatal()<<"Problème sur "<<InfoProc->me<<" de communication vers "
// 				 <<ToProc<<", lors de MPI_Isend";
//   }
  
//   return prequest;

// }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/** 
  Fonction qui se charge de diffuser un tableau (envois et réception).
  Il est fait un MPI\_Bcast mais il faut que la taille du tableau soit connue sur tous les processeurs.

  {\em Remarque:} La taille du tableau est en octets.

  @memo    Diffusion d'un message.
  @param   InfoProc  (I) structure décrivant le processeur sur lequel tourne l'application.
  @param   FromProc  (I) noeud de calcul depuis lequel on diffuse le tableau.
  @param   TabTMP    (I) tableau qui est envoyé.
  @param   TailleTMP (I) taille de TabTMP (en octets).
  @return  (void *) TabTMP : tableau qui est reçu.
  @see     Equilibrage
  @author  Eric Brière de l'Isle, ONERA, DRIS/SRL
  @version Création Octobre 1998
*/
void* SplitSDMeshPartitioner::DiffuseMessage(StrucInfoProc* InfoProc, int FromProc, void* TabTMP, int TailleTMP)
{
  int ierror; /* retour d'erreur sur les fonctions MPI */

  if (TailleTMP<=0) {
    InfoProc->m_service->pfatal()<<"DiffuseMessage depuis "<<InfoProc->me<<", il est nécessaire que la taille du tableau soit connue !!!\n";
  }


  if (InfoProc->me != FromProc)
    {
      if (TabTMP == NULL)
	{
	  TabTMP = malloc ((size_t) TailleTMP);
	  CHECK_IF_NOT_NULL(TabTMP);
	}
    } /* end if me != FromProc */
  

  /* diffusion du tableau TabTMP */
  ierror = MPI_Bcast (TabTMP,
		      TailleTMP,
		      MPI_PACKED,
		      FromProc,
		      InfoProc->Split_Comm);
  
  if (ierror != MPI_SUCCESS) {
    InfoProc->m_service->pfatal()<<"Problème sur "<<InfoProc->me<<" depuis "
				 <<FromProc<<", lors de MPI_Bcast";
  }

  return (TabTMP);

}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \brief Donne la taille en octets nécessaire pour le stockage mémoire d'un domaine
  (sans la liste des noeuds sur l'interface) en vue d'une communication.

  @memo    Donne la taille en octets d'une StructureBlocEtendu
  @param   Domaine (I) StructureBlocEtendu
  @return  (int) : la taille en octets
  @see     PackDom, UnpackDom
  @author  Eric Brière de l'Isle, Novembre 2007
*/
Integer SplitSDMeshPartitioner::
TailleDom(StructureBlocEtendu* Domaine)
{
  size_t s = (2+2*Domaine->NbIntf)*sizeof(int)+sizeof(double);
  return arcaneCheckArraySize(s);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/** 
  Stocke le contenu d'un Domaine dans un tableau en vue d'une communication (sans la liste des noeuds sur l'interface)

  @memo    Stockage d'une StructureBlocEtendu dans un tableau.
  @param   Domaine (I) StructureBlocEtendu
  @param   TabTMP (I/O) tableau que l'on remplit
  @param   TailleTMP (I) taille totale du tableau TabTMP
  @param   comm (I) environnement de communication
  @return  void
  @see     TailleDom, UnpackDom
  @author  Eric Brière de l'Isle, Novembre 2007
*/
void SplitSDMeshPartitioner::
PackDom(StrucInfoProc* InfoProc, StructureBlocEtendu* Domaine, void* TabTMP,
        int TailleTMP, MPI_Comm comm)
{
  int position = 0; // initialisation pour mettre dans le tableau au début
  int ier;

  ier = MPI_Pack(&Domaine->NbElements, 1, MPI_INT, TabTMP, TailleTMP, &position, comm);
  CHECK_MPI_PACK_ERR(ier);
  
  ier = MPI_Pack(&Domaine->PoidsDom, 1, MPI_DOUBLE, TabTMP, TailleTMP, &position, comm);
  CHECK_MPI_PACK_ERR(ier);

  ier = MPI_Pack(&Domaine->NbIntf, 1, MPI_INT, TabTMP, TailleTMP, &position, comm);
  CHECK_MPI_PACK_ERR(ier);

  for (int i=0; i<Domaine->NbIntf; i++){
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
  Extrait un Domaine d'un tableau en provenance d'une communication.

  @memo    Extraction des données pour renseigner une StrucListeDomMail
  @param   TabTMP (I) tableau d'où sont extraites les informations
  @param   TailleTMP (I) taille totale du tableau TabTMP
  @param   comm (I) environnement de communication pour la librairie
  @param   DomMail (I/O) StrucListeDomMail
  @return  void
  @see     TailleDom, PackDom
  @author  Eric Brière de l'Isle, Novembre 2007
*/
void SplitSDMeshPartitioner::UnpackDom(void* TabTMP, int TailleTMP, MPI_Comm comm, StrucListeDomMail* DomMail)
{
  int position = 0; // initialisation pour prendre au début du tableau

  MPI_Unpack(TabTMP, TailleTMP, &position, &DomMail->NbElements, 1, MPI_INT, comm);
  MPI_Unpack(TabTMP, TailleTMP, &position, &DomMail->Poids,      1, MPI_DOUBLE, comm);
  MPI_Unpack(TabTMP, TailleTMP, &position, &DomMail->NbVoisins,  1, MPI_INT, comm);
  
  DomMail->ListeVoisins = new StrucListeVoisMail[DomMail->NbVoisins];

  for (int i=0; i<DomMail->NbVoisins; i++){
    MPI_Unpack(TabTMP, TailleTMP, &position, &DomMail->ListeVoisins[i].NoDomVois, 1, MPI_INT, comm);
    MPI_Unpack(TabTMP, TailleTMP, &position, &DomMail->ListeVoisins[i].NbNoeudsInterface, 1, MPI_INT, comm);
    DomMail->ListeVoisins[i].Delta = 0.0;
  }  
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/** 
  Donne la taille en octets nécessaire pour le stockage mémoire d'un entier pour le domaine chargé, un autre entier pour le domaine voisin, et un double pour le Delta (transfert en poids)
*/

int SplitSDMeshPartitioner::TailleEquil()
{
  return 2*sizeof(int) + sizeof(double);
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/** 
  Stocke les 2 entiers et le double 
*/
void SplitSDMeshPartitioner::PackEquil(StrucInfoProc* InfoProc, int indDomCharge, int indDomVois, double Delta, void* TabTMP, int TailleTMP, MPI_Comm comm)
{
  int position = 0; // initialisation pour mettre dans le tableau au début
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
  Extrait les 2 entiers et le double  d'un tableau en provenance d'une communication.
*/
void SplitSDMeshPartitioner::UnpackEquil(void* TabTMP, int TailleTMP, MPI_Comm comm, int* indDomCharge, int* indDomVois, double* Delta)
{
  int position = 0; // initialisation pour prendre au début du tableau
  MPI_Unpack(TabTMP, TailleTMP, &position, indDomCharge, 1, MPI_INT, comm);
  MPI_Unpack(TabTMP, TailleTMP, &position, indDomVois, 1, MPI_INT, comm);
  MPI_Unpack(TabTMP, TailleTMP, &position, Delta, 1, MPI_DOUBLE, comm);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(SplitSDMeshPartitioner,
                        ServiceProperty("SplitSD",ST_SubDomain),
                        ARCANE_SERVICE_INTERFACE(IMeshPartitioner),
                        ARCANE_SERVICE_INTERFACE(IMeshPartitionerBase));

ARCANE_REGISTER_SERVICE_SPLITSDMESHPARTITIONER(SplitSD,SplitSDMeshPartitioner);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
