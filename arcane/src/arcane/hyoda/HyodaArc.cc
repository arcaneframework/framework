// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*****************************************************************************
 * Hyoda.cc                                                    (C) 2000-2012 *
 *                                                                           *
 * Service de debugger hybrid.                                               *
 *****************************************************************************/
#include "arcane/IMesh.h"
#include "arcane/IApplication.h"
#include "arcane/IParallelMng.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/FactoryService.h"
#include "arcane/ServiceFinder2.h"
#include "arcane/SharedVariable.h"
#include "arcane/CommonVariables.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/AbstractService.h"
#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/IOnlineDebuggerService.h"
#include "arcane/ITransferValuesParallelOperation.h"
#include "arcane/ServiceBuilder.h"

#include "arcane/hyoda/HyodaArc.h"
#include "arcane/hyoda/HyodaTcp.h"
#include "arcane/hyoda/HyodaIceT.h"
#include "arcane/hyoda/HyodaMatrix.h"
#include "arcane/hyoda/HyodaPapi.h"

#include <unistd.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/



/******************************************************************************
 * Hyoda SERVICE
 *****************************************************************************/
ARCANE_REGISTER_APPLICATION_FACTORY(Hyoda, IOnlineDebuggerService, Hyoda);

# define SWAP(n) (((n)<<24)|(((n)&0xff00)<<8)|(((n)>>8)&0xff00)|((n)>>24))

/******************************************************************************
 * Hyoda Constructor
 *****************************************************************************/
Hyoda::Hyoda(const ServiceBuildInfo& sbi): AbstractService(sbi),
                                           m_break_at_startup(false),
                                           m_configured(false),
                                           m_init_configured(false),
                                           m_gdbserver_rank(0),
                                           m_qhyoda_hooked(0.),
//#warning Hardcoded values must be sync'ed with QHyodaJob::QHyodaJob's ones
                                           m_qhyoda_adrs(0x100007Ful),
// L'adresse IP est settée par QHyoda, par défault 127.0.0.1
                                           m_qhyoda_port(3889),
// Le numéro du port est settée par QHyoda, par défaut 3889
                                           m_qhyoda_pyld(8*1024),
// Le payload à utiliser
                                           m_qhyoda_width_height(0x04000300ul),                                           
                                           m_target_cell_uid(0),
                                           m_data(NULL),
                                           m_ice_mesh(NULL),
                                           m_ice_matrix(NULL),
                                           m_tcp(NULL),
                                           m_variables_names(NULL),
                                           m_application(sbi.application()),
                                           m_papi(new HyodaPapi(this, sbi.application(),traceMng())),
                                           m_matrix_render(false){
  debug()<<"Hooking Hyoda";  
  if ((m_data=(hyoda_shared_data*)malloc(sizeof(hyoda_shared_data)))==NULL)
     throw Arcane::FatalErrorException(A_FUNCINFO,"Could not allocate data space for hook");
  m_data->global_iteration=0;
  m_data->global_time=0.;
  m_data->global_deltat=0.;
  m_data->global_cpu_time=0.;
  m_data->global_mesh_nb_cells=0;
  m_data->target_cell_uid=m_target_cell_uid;
  m_data->target_cell_rank=0;
  m_data->target_cell_nb_nodes=0;
  for(int i=0;i<HYODA_CELL_NB_NODES_MAX;++i)
    m_data->coords[i][0]=m_data->coords[i][1]=m_data->coords[i][2]=0.;
  
  // On test si l'on souhaite s'accrocher au démarrage
  if (!platform::getEnvironmentVariable("ARCANE_HYODA").null()) m_break_at_startup=true;
  if (!platform::getEnvironmentVariable("ARCANE_HYODA_MATRIX_RENDER").null()) m_matrix_render=true;

  if (m_break_at_startup) debug()<<"\33[7m[Hyoda] m_break_at_startup == Single shot!\33[m";
  
  // On récupère le host depuis lequel est lancé Hyoda
  String hyoda_host;
  if (!(hyoda_host=platform::getEnvironmentVariable("ARCANE_HYODA_HOST")).null()){
    debug()<<"\33[7m[Hyoda] hyoda_host="<<hyoda_host<<"\33[m";
  }

  // On récupère l'adresse du host depuis lequel est lancé Hyoda
  String hyoda_adrs;
  if (!(hyoda_adrs=platform::getEnvironmentVariable("ARCANE_HYODA_ADRS")).null()){
    UInt32 adrs=0;
    if (!builtInGetValue(adrs,hyoda_adrs)){
      //Attention: QHostAddress::toIPv4Address() = if the address is 127.0.0.1, the returned value is 2130706433 (i.e. 0x7f000001).
      // Alors que l'on souhaite 0x100007Ful => il faut donc swaper!
      m_qhyoda_adrs=SWAP(adrs);
      debug()<<"\33[7m[Hyoda] hyoda_adrs="<<m_qhyoda_adrs<<"\33[m";
    }
  }

  // Et son numéro de port
  String hyoda_port;
  if (!(hyoda_port=platform::getEnvironmentVariable("ARCANE_HYODA_PORT")).null()){
    Integer port=0;
    if (!builtInGetValue(port,hyoda_port)){
      m_qhyoda_port=port;
      debug()<<"\33[7m[Hyoda] hyoda_port="<<m_qhyoda_port<<"\33[m";
    }
  }
} 



/******************************************************************************
 * Hyoda Destructor
 *****************************************************************************/
Hyoda::~Hyoda(){
  debug()<<"\33[7m[Hyoda::~ Hyoda]\33[m";
  free(m_data);
  delete m_papi;
}



/******************************************************************************
 * Fonction de test pour savoir si j'ai la maille visée
 * Maillage initiale: UIDs + partitionnement => LIDs dont les ghosts
 * Le slider est bien sur le maillage initial, donc les UIDs et on cherche à quels LIDs
 * ils correspondent alors que les ghosts ont été incorporés
 *****************************************************************************/
LocalIdType Hyoda::targetCellIdToLocalId(ISubDomain *sd, UniqueIdType target_cell_uid){
  UniqueArray<UniqueIdType> uid(1);
  Int32UniqueArray lid(1);
  
  uid[0]=target_cell_uid;
  debug()<<"[Hyoda::targetCellIdToLocalId] uid="<<target_cell_uid;
  sd->defaultMesh()->itemFamily(IK_Cell)->itemsUniqueIdToLocalId(lid.view(),uid.constView(),false);
  debug()<<"[Hyoda::targetCellIdToLocalId] lid="<<lid[0];

  if (lid[0]!=NULL_ITEM_ID){ // Si on a un candidat: c'est une maille propre ou fantôme
    if (!sd->defaultMesh()->cellFamily()->itemsInternal()[lid[0]]->isOwn())
      return NULL_ITEM_ID; // Si c'est fantôme, c'est pas à nous
    return lid[0]; // Sinon, on la tient!
  }
  return NULL_ITEM_ID;
}



/******************************************************************************
 * Fonction de configuration pour mettre tout le monde d'accord sur les variables:
 * m_gdbserver_rank
 * m_data->*
 *****************************************************************************/
void Hyoda::broadcast_configuration(ISubDomain *sd, UniqueIdType target_cell_uid){
  debug()<<"[Hyoda::broadcast_configuration] target_cell_uid="<<target_cell_uid;
  m_configured=true;
  
  if (!sd->parallelMng()->isParallel()){
    // En séquentiel, il faut au moins mettre le nombre de mailles à jour
    m_data->global_mesh_nb_cells=sd->defaultMesh()->ownCells().size();
    // En séquentiel, il faut au moins mettre le nombre noeuds de la maille visée (uid=lid)
    m_data->target_cell_nb_nodes=sd->defaultMesh()->cellFamily()->itemsInternal()[target_cell_uid]->nbNode();
    // Si on est en mode séquentiel, il y a rien d'autre à faire
    return;
  }
  
  // Transformation unique => local
  LocalIdType target_cell_lid=targetCellIdToLocalId(sd, target_cell_uid);
  UniqueArray<Integer> gather(4*sd->parallelMng()->commSize());
  UniqueArray<Integer> all;
  // En mode parallèle, nous remplissons la structure data pour la partager
  if (m_break_at_startup==false)
    all.add(m_qhyoda_hooked>0.?3883:0);               // Où gdbserver est accroché
  else
    all.add(sd->parallelMng()->commRank()==0?3883:0); // Où gdbserver est accroché
  all.add(target_cell_lid);                           // Le résultat de la transformation id to 'local'
  all.add(target_cell_lid!=NULL_ITEM_ID               // Le nombre de noeuds de la maille considérée
          ?sd->defaultMesh()->cellFamily()->itemsInternal()[target_cell_lid]->nbNode():0);
  all.add(sd->defaultMesh()->ownCells().size());      // Le nombre de own cells par sous-domaine
  // On s'échange le tout /////////////////////////////////////////////
  debug()<<"[Hyoda::broadcast_where_gdbserver_is_hooked] Gathering, all="<<all<<"...";
  sd->parallelMng()->allGather(all,gather);
  ///////////////////////////////////////////////////////////////////// 
  m_gdbserver_rank=-1;
  m_data->target_cell_rank=-1;
  m_data->target_cell_nb_nodes=-1;
  m_data->global_mesh_nb_cells=0;
  for(int iRnk=4*sd->parallelMng()->commSize();iRnk>0;iRnk-=4){
    if (gather[iRnk-4]==3883){
      if (m_gdbserver_rank!=-1)
        throw FatalErrorException("[Hyoda::broadcast_where_gdbserver_is_hooked] more than one m_gdbserver_rank");
      m_gdbserver_rank=(iRnk/4)-1;
      debug()<<"[Hyoda::broadcast_where_gdbserver_is_hooked] m_gdbserver_rank="<<m_gdbserver_rank;
    }
    if (gather[iRnk-3]!=NULL_ITEM_ID){
      if (m_data->target_cell_rank!=-1)
        throw FatalErrorException("[Hyoda::broadcast_where_gdbserver_is_hooked] more than one m_data->target_cell_rank");
      m_data->target_cell_rank=(iRnk/4)-1;
      debug() << "\33[7m[Hyoda::broadcast_where_gdbserver_is_hooked] m_data->target_cell_rank="
             << m_data->target_cell_rank
             << "\33[m";
    }
    if (gather[iRnk-2]!=0){
      if (m_data->target_cell_nb_nodes!=-1)
        throw FatalErrorException("[Hyoda::broadcast_where_gdbserver_is_hooked] more than one m_data->target_cell_nb_nodes");
      m_data->target_cell_nb_nodes=gather[iRnk-2];
      debug()<<"[Hyoda::broadcast_where_gdbserver_is_hooked] m_data->target_cell_nb_nodes="<<m_data->target_cell_nb_nodes;
    }
    if (gather[iRnk-1]!=0) m_data->global_mesh_nb_cells+=gather[iRnk-1];
  }
  if (m_data->target_cell_rank>=sd->parallelMng()->commSize())
    throw FatalErrorException("[Hyoda::broadcast_where_gdbserver_is_hooked] ERROR with m_data->target_cell_rank (>commSize)");
}


/******************************************************************************
 * Vérifie si un sous-domaine demande un breakpoint ou une tâche particulière
 * On doit être rapide car c'est ce que l'on fait en mode standard
 * C'est le premier breapoint sur lequel gdb va s'arréter
 * Loopbreak pour positionner des points d'arrêt lors de la boucle en temps
 *****************************************************************************
 * Attention à l'ordre dans lequel nous allons returner
 *****************************************************************************/
Real Hyoda::
loopbreak(ISubDomain* sd)
{
  ARCANE_UNUSED(sd);
  // Si l'un voit la target cell uid bouger,
  // il faut informer les autres via le biais du reduce qui sera fait
  if (m_target_cell_uid!=m_data->target_cell_uid)
    return HYODA_HOOK_CONFIGURE;
  // Si on a l'IHM de up dès le début, on s'arréte systématiquement!
  if (m_break_at_startup)
    return HYODA_HOOK_BREAK;
  // Est-ce que mon m_qhyoda_hooked été set'à par gdb?
  if (m_qhyoda_hooked>0.)
    return HYODA_HOOK_BREAK;
  return 0.0;
}

/******************************************************************************
 * Softbreak pour positionner des points d'arrêt en différents endroits du code
 *****************************************************************************/
Real Hyoda::softbreak(ISubDomain* sd, const char *fileName,const char *prettyFunction, int lineNumber){
  // Tant que l'IHM n'est pas accrochée, on fait rien
  if (m_qhyoda_hooked==0.) return 0.;
  debug() << "[Hyoda::softbreak] Was @ " << ((fileName)?fileName:"(NoFileNameInfo)")<< ":" << lineNumber;
  debug() << "[Hyoda::softbreak] Was @ " << ((prettyFunction)?prettyFunction:"(NoFunctionInfo)");
  //usleep(100);
  return 0.;
}


/******************************************************************************
 * Point d'entrée du debugger
 * C'est le deuxième breapoint sur lequel gdb va s'arréter pour rafraîchir
 *****************************************************************************/
void Hyoda::hook(ISubDomain* sd, Real tasks){
  m_papi->stop();
  
  // On attend au moins d'avoir fait une itération
  //if (sd->commonVariables().globalIteration()<2) return;

  // Si la maille affichée dans la QHyodaToolCell est à rafraîchir,
  // on le demande && m_break_at_startup==false
  if (m_target_cell_uid!=m_data->target_cell_uid) tasks=HYODA_HOOK_CONFIGURE;

  // Dans le cas parallèle
  if (tasks==HYODA_HOOK_CONFIGURE && sd->parallelMng()->isParallel()){
    debug()<<"[Hyoda::hook] New requested cell #"<<m_target_cell_uid;
    UniqueArray<UniqueIdType> send_buf(1,m_target_cell_uid);
    sd->parallelMng()->broadcast(send_buf.view(),m_gdbserver_rank);
    debug() << "[Hyoda::hook] broadcasted cell " << send_buf.at(0)
           << ", m_data->global_mesh_nb_cells=" << m_data->global_mesh_nb_cells;
    if (send_buf.at(0)>m_data->global_mesh_nb_cells)
      throw FatalErrorException("[Hyoda::hook] ERROR with broadcasted cell");
    // Et on met à jour la bonne cell ID dans la structure vue du debugger
    m_data->target_cell_uid=m_target_cell_uid=send_buf.at(0);
    // Et on annonce qu'il faut se reconfigurer
    m_configured=false;
  }

  // Dans le cas séquentiel (pas sûr d'être utile)
  if (tasks==HYODA_HOOK_CONFIGURE && (!sd->parallelMng()->isParallel())){
    debug()<<"[Hyoda::hook (seq)] New cell sent from server "<<m_target_cell_uid;
    if (m_target_cell_uid>m_data->global_mesh_nb_cells)
      throw FatalErrorException("[Hyoda::hook] ERROR with broadcasted cell");
    // Et on met à jour la bonne cell ID dans la structure vue du debugger
    m_data->target_cell_uid=m_target_cell_uid;
    // Et on annonce qu'il faut se reconfigurer
    m_configured=false;
  }
 
  // S'il y a des configurations à faire, on les déclenche
  if (!m_configured) broadcast_configuration(sd, m_target_cell_uid);
    
  // Configuration à ne faire qu'une seule fois, après celle ci dessus pour que m_gdbserver_rank soit à jour
  if (!m_init_configured){
    m_init_configured=true;
    UniqueArray<Integer> wxh_to_broadcast(0);
    wxh_to_broadcast.add(m_qhyoda_width_height);
    debug()<<"[Hyoda::hook] broadcasting wxh...";
    sd->parallelMng()->broadcast(wxh_to_broadcast.view(), m_gdbserver_rank);
    unsigned int width  = (wxh_to_broadcast[0]>>16)&0xFFFFul;
    unsigned int height = (wxh_to_broadcast[0])&0xFFFFul;
    debug() << "[Hyoda::hook] Let's configure: m_qhyoda_width_height="
           << wxh_to_broadcast[0] << ", " << width << "x" << height;
    debug()<<"\33[7m[Hyoda::hook] Initializing HyodaTcp\33[m";
    // Seul le proc 0 se connecte réellement
    m_tcp = new HyodaTcp(this, sd, traceMng(),
                         m_qhyoda_adrs, m_qhyoda_port,
                         m_qhyoda_pyld, m_break_at_startup);
    debug()<<"\33[7m[Hyoda::hook] Initializing MESH HyodaIceT\33[m";
    m_ice_mesh=new HyodaIceT(this, sd, traceMng(), width, height, m_tcp);
    debug()<<"\33[7m[Hyoda::hook] Initializing MATRIX HyodaIceT\33[m";
    m_ice_matrix=new HyodaMatrix(this, sd, traceMng(), width, height, m_tcp);
    m_papi->initialize(sd,m_tcp);
  }
  
  m_ice_mesh->render();
  if (m_matrix_render) m_ice_matrix->render();
  
  debug()<<"[Hyoda::hook] now fetch_and_fill_data_to_be_dumped";
  fetch_and_fill_data_to_be_dumped(sd,m_target_cell_uid);
  
  debug()<<"[Hyoda::hook] dump";
  m_papi->dump();
  debug()<<"[Hyoda::hook] done";
  m_papi->start();
}


// ****************************************************************************
// * ijval
// ****************************************************************************
void Hyoda::ijval(int cpu,int n, int *i, int *j, double *val){
  debug()<<"\33[7m[HyodaArc::ijval] cpu="<<cpu<<", n="<<n<<"\33[m";
  m_ice_matrix->setIJVal(cpu,n,i,j,val);
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

