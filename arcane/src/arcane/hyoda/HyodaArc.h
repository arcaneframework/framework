// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*****************************************************************************
 * Hyoda.h                                                     (C) 2000-2012 *
 *                                                                           *
 * Header du debugger hybrid.                                                *
 *****************************************************************************/
#ifndef ARCANE_HYODA_ARC_H
#define ARCANE_HYODA_ARC_H


/******************************************************************************
 * DEFINES
 * Commandes (Real) codes sur lesquelles on reduce::max pour récupérer ce qu'il faut faire
 *****************************************************************************/
#define HYODA_HOOK_BREAK 1.
#define HYODA_HOOK_CONFIGURE 2.

// Nombre max de noeuds d'une maille
#define HYODA_CELL_NB_NODES_MAX 12


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


/******************************************************************************
 * Structure de dialogue entre Arcane et Hyoda-GUI
 *****************************************************************************/
struct hyoda_shared_data{
  Int64 global_iteration;
  Real global_time;
  Real global_deltat;
  Real global_cpu_time;
  Int64 global_mesh_nb_cells;
  Int64 target_cell_uid;
  Int64 target_cell_rank;
  Int64 target_cell_nb_nodes;
  Real coords[HYODA_CELL_NB_NODES_MAX][3];
};


class HyodaIceT;
class HyodaMatrix;
class HyodaTcp;
class HyodaPapi;

/******************************************************************************
 * Hyoda CLASS
 *****************************************************************************/
class Hyoda: public AbstractService,
             public IOnlineDebuggerService{
public:
  Hyoda(const ServiceBuildInfo& sbi);
  virtual ~Hyoda();
  virtual Real loopbreak(ISubDomain*);
  virtual Real softbreak(ISubDomain*,const char*,const char*,int);
  virtual void hook(ISubDomain*,Real);
  virtual void ijval(int,int,int*,int*,double*);
  IApplication* application(){return m_application;}
  HyodaIceT *meshIceT(void){return m_ice_mesh;}
  //HyodaMatrix *matrixIceT(void){return m_ice_matrix;}
private:
  void fetch_and_fill_data_to_be_dumped(ISubDomain*, UniqueIdType);
  void broadcast_configuration(ISubDomain*, UniqueIdType);
private:
  LocalIdType targetCellIdToLocalId(ISubDomain *sd, UniqueIdType target_cell_id);
  //void icetHyodaInit(void);
private:
  //! Variable indiquant le mode 'single' et qu'il faut donc s'arrêter tout de suite
  bool m_break_at_startup;
  //! Variable indiquant si Hyoda et Arcane se sont configurés
  bool m_configured;
  bool m_init_configured;
  //! Rang sur lequel est accroché Hyoda
  Integer m_gdbserver_rank;
  //! Variable dans laquelle QHyoda annonce s'il est ou pas accroché
  Real m_qhyoda_hooked;
  //! Variable dans laquelle QHyoda renseigne l'adresse de la machine
  UInt32 m_qhyoda_adrs;
  //! Variable dans laquelle QHyoda renseigne le numéro de port à utiliser
  Integer m_qhyoda_port;
  //! Variable dans laquelle QHyoda renseigne le payload à utiliser
  Integer m_qhyoda_pyld;
  //! Variable dans laquelle QHyoda renseigne les dimensions à utiliser pour l'image
  Integer m_qhyoda_width_height;
  //! Variable dans laquelle QHyoda renseigne l'UID de la maille visée
  UniqueIdType m_target_cell_uid;
  //!< Variable pointant vers un tableau de noms de variables possible d'afficher 
  char **m_variables_names;
  //!< Variable pointant vers l'application pour éventuellement aller chercher les arguments
  IApplication* m_application;
  //!< structure à remplir avant que Hyoda ne la dump pour l'exploiter
  struct hyoda_shared_data *m_data;
private:
  HyodaIceT *m_ice_mesh;
  HyodaMatrix *m_ice_matrix;
  HyodaTcp *m_tcp;
  HyodaPapi *m_papi;
  bool m_matrix_render;
};


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  // ARCANE_HYODA_ARC_H
