// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SplitSDMeshPartitioner.h                                    (C) 2000-2024 */
/*                                                                           */
/* Partitioneur de maillage reprenant le fonctionnement de SplitSD.          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_FILES_SPLITSDMESHPARTITIONER_H
#define ARCANE_FILES_SPLITSDMESHPARTITIONER_H

// #define DEBUG_PARTITIONER

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#include "arcane/std/MeshPartitionerBase.h"

#include "arcane/std/SplitSDMeshPartitioner_axl.h"

#define MPICH_SKIP_MPICXX
#define OMPI_SKIP_MPICXX
#include <mpi.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#define TAG_MAILLAGEMAITRE        1

#define CHECK_MPI_PACK_ERR(ier) \
  if (ier != MPI_SUCCESS) { \
     switch (ier) { \
     case MPI_ERR_COMM:InfoProc->m_service->pfatal()<<"erreur sur MPI_Pack de type MPI_ERR_COMM"; break; \
     case MPI_ERR_TYPE: InfoProc->m_service->pfatal()<<"erreur sur MPI_Pack de type MPI_ERR_TYPE"; break; \
     case MPI_ERR_COUNT: InfoProc->m_service->pfatal()<<"erreur sur MPI_Pack de type MPI_ERR_COUNT"; break; \
     case MPI_ERR_ARG: InfoProc->m_service->pfatal()<<"erreur sur MPI_Pack de type MPI_ERR_ARG"; break; \
     default: InfoProc->m_service->pfatal()<<"erreur sur MPI_Pack de type inconnu !"; \
     } \
  }

#ifndef CHECK_IF_NOT_NULL
/** 
  Macros qui fait le test comme quoi le pointeur en argument n'est pas nul
  et qui fait appel à pfatal() si ce pointeur s'avère nul.
  Cette macro est à utiliser après tout appel à l'une des fonctions utilisant la mémoire
  (malloc, calloc, realloc ...).

  @memo   Teste si un pointeur n'est pas nul.

  @param  Ptr un pointeur

  @author  Eric Brière de l'Isle, ONERA, DRIS/SRL

*/
#define CHECK_IF_NOT_NULL(Ptr) \
  if (Ptr==NULL) { \
      InfoProc->m_service->pfatal()<<"Pointeur vaut nil, (On manque peut-etre de place memoire !!! )"; \
    }
#endif

#ifndef MAX
#define MAX(a,b) ((a)<(b)?(b):(a))
#endif
#ifndef MIN
#define MIN(a,b) ((a)>(b)?(b):(a))
#endif
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/**     Structure décrivant un processeur.
        @name  StrucInfoProc
*/
typedef struct T_InfoProc
{
/*@{*/
  /** Numéro du processeur, ce numéro est compris entre 0 et le nombre de processeurs moins un.*/
  int      me;
  /** Nombre total de processeurs, 
      c'est le nombre maximum de processeurs que cette application 
      peut utiliser lors d'une exécution.*/
  int      nbSubDomain;
//   /** Numéro du processeur maître, 
//       c'est sur ce processeur que sont faites l'agglomération, 
//       l'insertion et le redécoupage.*/
//   int      NoMaitre;
  /** Groupe de travail sous MPI 
      (la définition d'un nouveau groupe permet de limiter la porté 
      des messages utilisés).*/
  MPI_Comm              Split_Comm;

  /// Pour la gestion des traces (messages, erreurs, fatal ...)
  Arcane::AbstractService*  m_service;

/*@}*/
} StrucInfoProc;
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/** 
    Structure pour le stockage des informations locales, pour éviter de les recalculer plusieurs fois.
    Il s'agit essentiellement des noeuds sur les interfaces avec les différents domaines voisins.
    EB, 2007. Simplification de StructureBlocEtendu
*/


typedef struct T_INTERFACE
{
  /**     Elément permettant de décrire une interface avec un seul bloc voisin        
	  @name  StructureInterface
  */

  /** Numero du domaine voisin commence a 0	*/
  int                                 NoDomVois;
  /** Liste Des Noeuds interface		*/
  Arcane::UniqueArray<Arcane::Node>         ListeNoeuds;

} StructureInterface;


typedef struct T_BLOC_ETENDU
{
  /**     Elément permettant de décrire un bloc étendu et son environnement        
	  @name  StructureBlocEtendu
  */

  /**      Nombre de mailles */
  Integer NbElements;

  /**      Poids total du domaine */
  double                             PoidsDom;

  /**      Nombre d'interfaces  */
  Integer NbIntf;
  
  /**      Tableau de structures d'interfaces, une pour chacun des voisins du bloc en question  */
  StructureInterface*                Intf;
  
               
} StructureBlocEtendu;
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/**     Elément permettant de décrire une interface entre deux blocs.
        @name  StrucListeVoisMail
        @see   StrucListeDomMail         
*/
typedef struct T_ListeVoisMail
{
/*@{*/
  /** Numéro du domaine voisin (de 0 au nombre de processeurs moins un) */
  int                 NoDomVois;
  /** Nombre de noeuds dans l'interface commune */
  int                 NbNoeudsInterface;
  /** Quantité à transférer par cette interface */
  double              Delta;
  /** Poids pour l'interface */
//   int                 PoidsInterface;  pas de signification ici
/*@}*/
} StrucListeVoisMail;

/**     Elément pour décrire un bloc dans la description globale du maillage.
        @name  StrucListeDomMail
        @see  StrucMaillage
*/
typedef struct T_ListeDomMail
{
/*@{*/
  /** Nombre d'éléments dans le domaine */
  int                 NbElements;
  /** Poids pour le domaine */
  double              Poids;
  /** Nombre de blocs voisins */
  int                 NbVoisins;
  /** Liste de descripteurs pour les voisins */
  StrucListeVoisMail* ListeVoisins;
/*@}*/
} StrucListeDomMail;

/**     Structure permettant la description globale du maillage.\\

  {\em Remarque:} le stockage des numéros globaux inutilisés 
  est effectué sur le processeur maître, 
  le maître peut changer aussi est-ce transmis au nouveau 
  maître lors de la mise à jour du maillage maître (\Ref{MAJMaillageMaitre}).

  @name  StrucMaillage
  @see   main, MAJMaillageMaitre
*/
typedef struct T_Maillage
{
/*@{*/
  /** Nombre d'éléments dans le maillage global */
  int                 NbElements;
  /** Nombre d'éléments dans le maillage global */
  double              Poids;
 
  /** Nombre de blocs non vides */
  int                 NbDomainesPleins;
  /** Nombre de domaines total, c'est le nombre de processeurs attribués à cette session */
  int                 NbDomainesMax;
  /** Liste (de taille NbDomainesMax) de descripteurs de blocs */
  StrucListeDomMail*  ListeDomaines;

  /** Nombre de processeurs dont le sous-domaine est vide */
  int                 NbProcsVides;
//   /** Liste des processeurs sans sous-domaine */
//   int*                ListeNoProcsVides;

/*@}*/
} StrucMaillage;
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Partitioneur de maillage inspiré de la bibliothèque SplitSD, développé 
 *  initialement à l'ONERA pour Dassault Aviation.
 */
class SplitSDMeshPartitioner
: public ArcaneSplitSDMeshPartitionerObject
  //: public MeshPartitionerBase
{
 public:
  
  SplitSDMeshPartitioner(const ServiceBuildInfo& sbi);

 public:

  virtual void build() {}

 public:

  virtual void partitionMesh(bool initial_partition);
  virtual void partitionMesh(bool initial_partition,Int32 nb_part)
  {
    ARCANE_UNUSED(initial_partition);
    ARCANE_UNUSED(nb_part);
    throw NotImplementedException(A_FUNCINFO);
  }

 private:
  /// initialisation des structures
  void init(bool initial_partition, StrucInfoProc* &InfoProc, StructureBlocEtendu* &Domaine, StrucMaillage* &Maillage);

  /// initialisation des poids (m_cells_weight => m_poids_aux_mailles)
  void initPoids(bool initial_partition);

  /// libération mémoire des structures
  void fin (StrucInfoProc* &InfoProc, StructureBlocEtendu* &Domaine, StrucMaillage* &Maillage);

  /// mise à jour de la structure locale (noeuds sur les interfaces avec sous-domaines voisins)
  void MAJDomaine(StructureBlocEtendu* Domaine);

  /// mise à jour de la structure sur le processeur 0
  void MAJMaillageMaitre(StrucInfoProc* InfoProc, StructureBlocEtendu* Domaine, StrucMaillage* Maillage);

  /// fonction de vérification de la cohérence (réciprocité) des interfaces
  void verifMaillageMaitre(StrucMaillage* Maillage);

  /// On utilise une méthode de parcours frontal pour aller d'un noeud surchargé vers les autres noeuds en mémorisant le chemin pour mettre à jour les Delta sur les interfaces.
  void MAJDeltaGlobal(StrucInfoProc* InfoProc, StrucMaillage* Maillage, double tolerance);
  /// fonction de décalage du Delta associé à une interface recherchée pour un couple, domaine et numéro de voisin, spécifié 
  void MAJDelta(double don, int iDOmTmpPrec,int iDomTmp, StrucListeDomMail* ListeDomaines);

  /// calcul un deltaMin en fonction des transferts locaux
  double CalculDeltaMin(StrucMaillage* Maillage, double deltaMin, int iterEquilibrage, int NbMaxIterEquil);

  /// phase itérative pour équilibrer la charge
  void Equilibrage(StrucInfoProc* InfoProc, StructureBlocEtendu* Domaine, StrucMaillage* Maillage);

  /// phase de transfert entre 2 domaines, MAJ des domaines
  void Equil2Dom(int* MasqueDesNoeuds, int* MasqueDesElements, int marqueVu, int marqueNonVu,
		 StrucInfoProc* InfoProc, StructureBlocEtendu* Domaine, StrucMaillage* Maillage, 
		 int indDomCharge, int indDomVois, double Delta);

  /// sélection d'éléments dans un domaine pour équilibrage entre 2 dom, en faisant un parcour frontal depuis l'interface
  void SelectElements(int* MasqueDesNoeuds, int* MasqueDesElements, int marqueVu, int marqueNonVu,
		      StrucInfoProc* InfoProc, StructureBlocEtendu* Domaine, 
		      double Delta, int indDomVois, Arcane::Array<Arcane::Cell>& ListeElements);

  /// parcours frontal limité suivant le Delta (poids cumulés des éléments pris dans les fronts), retourne un en cas de blocage
  int ParcoursFrontalDelta (int* MasqueDesNoeuds, int* MasqueDesElements, 
			    int marqueVu, int marqueNonVu,
			    double Delta,
			    int *pNbFronts, int NbFrontsMax,
			    Arcane::Array<Arcane::Node>& FrontsNoeuds,  int* IndFrontsNoeuds,
			    Arcane::Array<Arcane::Cell>& FrontsElements,int* IndFrontsElements);
  
  /** lissage du dernier front obtenu par parcours frontal, 
      de manière à intégrer dans ce front les éléments dont tous les noeuds
      sont déjà pris dans les fronts précédents
  */
  void LissageDuFront (int* MasqueDesNoeuds, int* MasqueDesElements, 
		       int marqueVu, int marqueNonVu,
		       int NbFronts,
		       Arcane::Array<Arcane::Node>& FrontsNoeuds,  int* IndFrontsNoeuds,
		       Arcane::Array<Arcane::Cell>& FrontsElements,int* IndFrontsElements);

  /// déplace des parties du sous-domaine lorsqu'elles sont trop petites et non connexes
  void ConnexifieDomaine(StrucInfoProc* InfoProc, StructureBlocEtendu* Domaine, StrucMaillage* Maillage, double tolConnexite);

  /// recherche le domaine voisin ayant le max de Faces en commun avec le groupe de mailles
  int getDomVoisMaxFace(Arcane::Array<Arcane::Cell>& ListeElements, int me);

  /// création d'un tableau qui sert de masque sur les LocalId des noeuds
  int* GetMasqueDesNoeuds(StrucInfoProc* InfoProc);
  /// création d'un tableau qui sert de masque sur les LocalId des éléments
  int* GetMasqueDesElements(StrucInfoProc* InfoProc);

  void LibereInfoProc(StrucInfoProc* &InfoProc);
  void LibereDomaine(StructureBlocEtendu* &Domaine);
  void LibereMaillage(StrucMaillage* &Maillage);

  void AfficheDomaine (int NbDom, StructureBlocEtendu* Domaine);
  void AfficheMaillage (StrucMaillage* Maillage);
  void AfficheListeDomaines(StrucListeDomMail* ListeDomaines, int NbDomaines);
  void AfficheEquilMaillage(StrucMaillage* Maillage);


  void* RecoitMessage (StrucInfoProc* InfoProc, int FromProc, int Tag, int *pTailleTMP);
  void  EnvoieMessage (StrucInfoProc* InfoProc, int ToProc,   int Tag, void* TabTMP, int TailleTMP);
//   MPI_Request*  EnvoieIMessage(StrucInfoProc* InfoProc, int ToProc, int Tag, void* TabTMP, int TailleTMP);
  void* DiffuseMessage(StrucInfoProc* InfoProc, int FromProc, void* TabTMP, int TailleTMP);
  
  // taille pour le transfert sans les noeuds, seulement la taille de ListeNoeuds
  int TailleDom(StructureBlocEtendu* Domaine);
  // les données sont mise dans un tableau
  void PackDom(StrucInfoProc* InfoProc, StructureBlocEtendu* Domaine, void* TabTMP, int TailleTMP, MPI_Comm comm);
  // les données sont extraite d'un tableau
  void UnpackDom(void* TabTMP, int TailleTMP, MPI_Comm comm, StrucListeDomMail* DomMail);
  
  // taille pour le transfert des numéros de domaine pour Equilibrage, et du Delta
  int TailleEquil();
  // les données sont mise dans un tableau
  void PackEquil(StrucInfoProc* InfoProc, int indDomCharge, int indDomVois, double Delta, void* TabTMP, int TailleTMP, MPI_Comm comm);
  // les données sont extraite d'un tableau
  void UnpackEquil(void* TabTMP, int TailleTMP, MPI_Comm comm, int* indDomCharge, int* indDomVois, double* Delta);
  

private:
  VariableCellReal m_poids_aux_mailles;    // poids que l'on calcul une fois par appel au rééquilibrage et qui suit les cellules
};
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
