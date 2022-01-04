// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HyodaEnvs.cc                                                (C) 2000-2013 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef _HYODA_PLUGIN_ENVIRONMENTS_H_
#define _HYODA_PLUGIN_ENVIRONMENTS_H_

#include "arcane/IApplication.h"
#include "arcane/IParallelMng.h"

#include "arcane/AbstractService.h"
#include "arcane/FactoryService.h"

#include "arcane/IVariableMng.h"
#include "arcane/SharedVariable.h"
#include "arcane/CommonVariables.h"

#include "arcane/IMesh.h"
#include "arcane/IItemFamily.h"

#include "arcane/materials/IMeshMaterialMng.h"
#include "arcane/materials/IMeshMaterial.h"
#include "arcane/materials/IMeshEnvironment.h"
#include "arcane/materials/MeshMaterialModifier.h"
#include "arcane/materials/MeshMaterialVariableRef.h"
#include "arcane/materials/MaterialVariableBuildInfo.h"
#include "arcane/materials/MeshEnvironmentBuildInfo.h"
#include "arcane/materials/CellToAllEnvCellConverter.h"
#include "arcane/materials/MatItemVector.h"

#include "arcane/hyoda/Hyoda.h"
#include "arcane/hyoda/HyodaArc.h"
#include "arcane/hyoda/HyodaMix.h"
#include "arcane/hyoda/HyodaIceT.h"
#include "arcane/hyoda/IHyodaPlugin.h"

//#include <GL/osmesa.h>
#include "GL/glu.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
class HyodaMix;
using namespace Arcane;
using namespace Arcane::Materials;

// *****************************************************************************
// * DEFINES
// *****************************************************************************
#define HYODA_MAX_ENV 2


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
class HyodaEnvs: public AbstractService, public IHyodaPlugin{
public:
  // **************************************************************************
  // * Arcane Service HyodaEnvs
  // **************************************************************************
  HyodaEnvs(const ServiceBuildInfo& sbi):
    AbstractService(sbi),
    m_sub_domain(sbi.subDomain()),
    m_default_mesh(m_sub_domain->defaultMesh()),
    m_material_mng(IMeshMaterialMng::getReference(m_default_mesh)),
    m_interface_normal(VariableBuildInfo(m_default_mesh,
                                         "InterfaceNormal")),
    m_interface_distance_env(VariableBuildInfo(m_default_mesh,
                                               "InterfaceDistance2ForEnv",
                                               IVariable::PNoDump|IVariable::PNoRestore))
  {
    info() << "Loading Hyoda's environments plugin";
    info() << "Setting a maximum of " << HYODA_MAX_ENV << " environments";
  }

  
  // **************************************************************************
  // * drawGlobalCell
  // **************************************************************************
  int drawGlobalCell(Cell cell, Real min, Real max, Real val){
    Real3 rgb;
    glBegin(GL_POLYGON);
    hyoda()->meshIceT()->setColor(min,max,val,rgb);
    glColor3d(rgb[0], rgb[1], rgb[2]);
    ENUMERATE_NODE(node, cell->nodes()){
      glColor3d(rgb[0], rgb[1], rgb[2]);
      glVertex2d(m_default_mesh->nodesCoordinates()[node].x,
                 m_default_mesh->nodesCoordinates()[node].y);
    }
    glEnd();
    // On repasse pour dessiner les contours de la maille
    glBegin(GL_QUADS);
    glColor3d(0.0, 0.0, 0.0);
    ENUMERATE_NODE(node, cell->nodes()){
      glVertex2d(m_default_mesh->nodesCoordinates()[node].x,
                 m_default_mesh->nodesCoordinates()[node].y);
    }
    glEnd();
    return 0;
  }


  // **************************************************************************
  // * computeDistMinMax
  // **************************************************************************
  void computeDistMinMax(Cell cell, Real3 normal, Real &dist_min, Real &dist_max){
    // Détermination de l'intervalle de recherche pour le Newton  
    dist_min = math::scaMul(m_default_mesh->nodesCoordinates()[cell.node(0)],normal);
    dist_max =math::scaMul(m_default_mesh->nodesCoordinates()[cell.node(0)],normal);
    const Integer& cell_nb_node=cell.nbNode();
    for (Integer node_id = 0; node_id < cell_nb_node; ++node_id) {
		const Node& node=cell.node(node_id);
		Real dist = math::scaMul(m_default_mesh->nodesCoordinates()[node],normal);
		if (dist < dist_min) dist_min = dist;  // premier sommet d'une maille quelconque
		if (dist > dist_max) dist_max = dist;  // dernier sommet d'une maille quelconque
    }
  }

  
  // **************************************************************************
  // * draw
  // **************************************************************************
  int draw(IVariable *variable, Real global_min, Real global_max)
  {
    ARCANE_UNUSED(global_min);
    ARCANE_UNUSED(global_max);
    // On ne supporte pour l'instant que l'affichage pour des variables aux mailles
    // Le test se fait au sein de HyodaMix::xLine2Cell
    info()<<"\n\r\33[7m[HyodaEnvs::draw] Focusing on variable "<<variable->name()<< "\33[m";
    MaterialVariableCellReal cell_variable(MaterialVariableBuildInfo(m_material_mng,variable->name()));
    MaterialVariableCellInt32 order_env(MaterialVariableBuildInfo(m_material_mng, "OrderEnv"));
    VariableCellArrayInteger number_env(VariableBuildInfo(m_default_mesh,"NumberEnv"));
    VariableCellInteger m_x_codes(VariableBuildInfo(m_default_mesh,"IntersectionCodes"));
    
    // Calcul des min et max locaux aux envCells
    debug()<<"\33[7m[HyodaEnvs::draw] Calcul des min et max locaux aux envCells\33[m";
    Real local_min, local_max;
    local_min=local_max=0.0;
    ENUMERATE_ALLENVCELL(iAllEnvCell, m_material_mng, m_default_mesh->allCells()){
      AllEnvCell allEnvCell = *iAllEnvCell;
      Cell cell=allEnvCell.globalCell();
      for(int iEnvOrder=0, mx=allEnvCell.nbEnvironment(); iEnvOrder<mx; ++iEnvOrder){
        Integer num_env = number_env[cell][iEnvOrder];
        if (num_env==-1) continue;
        ENUMERATE_CELL_ENVCELL(iEnvCell,allEnvCell){
          EnvCell envCell = *iEnvCell;
          if (num_env!=envCell.environmentId()) continue;
          Real val=cell_variable[envCell];
          local_min = math::min(local_min,val);
          local_max = math::max(local_max,val);
          break;
        } // ENUMERATE_CELL_ENVCELL
      } // for
    } // ENUMERATE_ALLENVCELL
    local_min=m_sub_domain->parallelMng()->reduce(Parallel::ReduceMin, local_min);
    local_max=m_sub_domain->parallelMng()->reduce(Parallel::ReduceMax, local_max);

    
    debug()<<"\33[7m[HyodaEnvs::draw] ENUMERATE_ALLENVCELL\33[m";
    ENUMERATE_ALLENVCELL(iAllEnvCell, m_material_mng, m_default_mesh->allCells()){
      AllEnvCell allEnvCell = *iAllEnvCell;
      Cell cell=allEnvCell.globalCell();
      Real3 normal = m_interface_normal[cell];
      Int32UniqueArray xCodes(0);
      
      // Si c'est pas initialisé, on se retrouve avec normal=(0,0,0)
      if (normal.abs()==0.) continue;
      info()<<"\n\r\33[7m[HyodaEnvs::draw] #"<<cell.uniqueId()<<", normal="<<normal << "\33[m";
 
      // On a qu'une seule origine par maille, on la calcule
      hyodaMix()->setCellOrigin(cell);
      m_x_codes[cell]=0;
      Real dist_min, dist_max;
      
      // On calcule les distance min et max pour être dans la maille
      computeDistMinMax(cell,normal,dist_min,dist_max);

      // On compte le nombre de milieux
      Int32 nbMilieux=0;
      ENUMERATE_CELL_ENVCELL(iEnvCell,allEnvCell) nbMilieux+=1;

      // On compte le nombre de milieux trouvés à partir des number_env
      Int32 nbNumberedMilieux=0;
      for(int iEnvOrder=0, mx=allEnvCell.nbEnvironment(); iEnvOrder<mx; ++iEnvOrder){
        Integer num_env = number_env[cell][iEnvOrder];
        if (num_env==-1) continue;
        nbNumberedMilieux+=1;
      }
      
      // On compte le nombre de milieux trouvés à partir des order_env
      Int32 nbOrderedMilieux=0;
      ENUMERATE_CELL_ENVCELL(iEnvCell,allEnvCell){
        Integer ord_env = order_env[iEnvCell];
        if (ord_env==-1) continue;
        if (ord_env > HYODA_MAX_ENV)
          warning() << "\33[7m[HyodaEnvs::draw] #"<<cell.uniqueId()
                    << ": while counting nbOrderedMilieux,  ord_env="
                    << ord_env << " > " << HYODA_MAX_ENV << "\33[m";
        if (ord_env > HYODA_MAX_ENV) continue;
        nbOrderedMilieux+=1;
      }

      
      // Focus sur une maille
      //if (cell.uniqueId()==49){
      {
        //info()<< "\n\33[7m[HyodaEnvs::draw] Cell "<<cell.uniqueId()<<"\33[m";
        //info()<< "[HyodaEnvs::draw] nbMilieux " << nbMilieux;
        //info()<< "[HyodaEnvs::draw] nbNumberedMilieux "<< nbNumberedMilieux;
        //info()<< "[HyodaEnvs::draw] nbOrderedMilieux " << nbOrderedMilieux;
        /*ENUMERATE_CELL_ENVCELL(iEnvCell,allEnvCell){
          Integer ord_env = order_env[iEnvCell];
          if (ord_env>nbMilieux){
            warning()<<"ord_env="<<ord_env<<" > nbMilieux="<<nbMilieux;
            ord_env=nbMilieux-1;
            //if (nbMilieux==1) ord_env=0; else fatal()<<"Cannot fix ord_env>nbMilieux";
          }
          info()<<"\t[HyodaEnvs::draw] ord_env="<<ord_env;
          info()<<"\t[HyodaEnvs::draw] number_env[cell][ord_env]="<<number_env[cell][ord_env];
          }*/
      }

      Int32 iMilieux=0;
      for(int iEnvOrder=0, mx=allEnvCell.nbEnvironment(); iEnvOrder<mx; ++iEnvOrder){
        Integer num_env = number_env[cell][iEnvOrder];
        if (num_env==-1) continue;
        Real its_distance=m_interface_distance_env[cell][num_env];
        info() << "[HyodaEnvs::draw] num_env=" << num_env << ", its distance=" << its_distance;
        if (its_distance < dist_min) warning()<<"its_distance < dist_min";
        if (its_distance > dist_max) warning()<<"its_distance > dist_max";
        // Il faut faire le xCellPoints pour setter les x[i]
        Int32 xPts=hyodaMix()->xCellPoints(cell, normal, its_distance, /*order=*/iMilieux);
        // Il faut traiter le cas < dist_min: on set le code à 0x0
        if (its_distance < dist_min) xPts=0x0;
        // Il faut traiter le cas > dist_max: on set le code à 0xF
        if (its_distance > dist_max) xPts=0xF;
        // Cas où on a qu'un seul milieu
        if (nbNumberedMilieux==1) xPts=0xF;
        info() << "\t\33[7m[HyodaEnvs::draw] xPts["<<iEnvOrder<<"]="<<xPts<< "\33[m";
        // Dans tous les cas, on l'ajoute pour pouvoir faire la distinction apres entre les x[i[0~11]]
        // On sauve les codes
        m_x_codes[cell]|=xPts<<(iMilieux<<2);
        xCodes.add(xPts);
        //if ((xPts!=0) && (xPts!=0xF)) hyodaMix()->xCellDrawNormal(cell, iOrg, its_distance);
        // On doit incrémenter iMilieux, c'est lui qui nous permet de trouver les bons x[i]
        iMilieux+=1;
      }
      
      //if (cell.uniqueId()==49)
      //info() << "[HyodaEnvs::draw] \33[7m#" << cell.uniqueId() << "\t" << xCodes<< "\33[m";
      //<< ", m_x_codes[this]=" << m_x_codes[cell] << "\33[m";
 
      // Pour chaque milieu, on revient pour remplir dans l'ordre
      // On fait dans ce sens (for, ENUMERATE_CELL_ENVCELL & break) pour éviter de jouer
      // avec order_env qui a tendance à être > nbMilieux
      // Mais dans ce sens, il y a des cas où l'on obtient pas (iMilieux==nbMilieux)
      // du fait de la non correspondance des number_env et environmentId()
      iMilieux=0;
      for(int iEnvOrder=0, mx=allEnvCell.nbEnvironment(); iEnvOrder<mx; ++iEnvOrder){
        Integer hit_env=-1;
        Integer num_env = number_env[cell][iEnvOrder];
        if (num_env==-1) continue;
        debug()<<"\t[HyodaEnvs::draw] Looking for num_env="<<num_env;
        // On déclenche les remplissage
        ENUMERATE_CELL_ENVCELL(iEnvCell,allEnvCell){
          EnvCell envCell = *iEnvCell;
          //if (cell.uniqueId()==49)
          debug()<<"\t\t[HyodaEnvs::draw] envCell.environmentId()="<<envCell.environmentId();
          if (num_env!=envCell.environmentId()) continue;
          //if (cell.uniqueId()==49){
          {
            debug() << "\t\t[HyodaEnvs::draw] \33[7m"
                   << iMilieux << "/" << (nbNumberedMilieux-1)
                   << " val=" << cell_variable[envCell] << "\33[m";
            debug()<<"\t\t[HyodaEnvs::draw] iMilieux="<<iMilieux;
          }
          hyodaMix()->xCellFill(cell, xCodes,
                                local_min, local_max,
                                cell_variable[envCell],
                                /*order=*/iMilieux,
                                nbNumberedMilieux);
          // On dit que l'on a hité le bon environment
          hit_env=num_env;
          iMilieux+=1;
          // On break car on a trouvé notre bonne envCell
          break;
        } // ENUMERATE_CELL_ENVCELL
        // Si on a pas trouvé le bon environment, on break pour essayer l'autre methode
        if (hit_env!=num_env){
          warning()<< "\33[7m[HyodaEnvs::draw] #"<<cell.uniqueId()
                   <<": Environement "<<num_env<<" non trouvé dans les milieux!\33[m";
          break;
        }
      } // for

      // Si on a pas réussi à tout dessiner, on revient en passant par les order_env
      // en esperant ne pas tomber sur les cas où les orders sont > HYODA_MAX_ENV
      // Cas i23.plt à l'iteration 186
      if (iMilieux!=nbNumberedMilieux){
        warning()<< "\33[7m[HyodaEnvs::draw] #"<<cell.uniqueId()
                   <<": tous les milieux n'ont pas été dessinés!\33[m";
        Integer iOrderedMilieux=0;
        Integer iPatchOffset=0;
        ENUMERATE_CELL_ENVCELL(ienvcell,allEnvCell){
          EnvCell envCell = *ienvcell;
          Integer order = order_env[ienvcell];
          if (order == -1 ) continue;
          if (order > HYODA_MAX_ENV){
            warning() << "\33[7m[HyodaEnvs::draw] #"<<cell.uniqueId()
                      <<": order_env > "<<HYODA_MAX_ENV<<"\33[m";
            continue;
          }
          if (order>=nbOrderedMilieux){
            warning() << "\33[7m[HyodaEnvs::draw] #"<<cell.uniqueId()
                      << ": En redessinant via order_env, le milieu "<<order
                      << " est réclamé, alors qu'on en a "<<nbOrderedMilieux<<"\33[m";
            warning() << "\33[7m[HyodaEnvs::draw] #"<<cell.uniqueId()
                      << ": On tente de patcher l'offset\33[m";
            iPatchOffset=1;
          }
          // On vérifie qu'on ne tombe pas négatif
          if (order<iPatchOffset) continue;
          hyodaMix()->xCellFill(cell, xCodes,
                                local_min, local_max,
                                cell_variable[envCell],
                                order-iPatchOffset,
                                nbOrderedMilieux);
          iOrderedMilieux+=1;
        }
        // Cette fois ci, on ne peut plus rien faire
        if (iOrderedMilieux!=nbOrderedMilieux)
          warning() << "\33[7m[HyodaEnvs::draw] #"<<cell.uniqueId()
                    << ": Même via les order_env, les milieux n'ont pas tous été dessinés\33[m";
      }
      
    } // ENUMERATE_ALLENVCELL
    return 0;
  }

  
  private:
  ISubDomain *m_sub_domain;
  IMesh* m_default_mesh;
  IMeshMaterialMng *m_material_mng ;
  VariableCellReal3 m_interface_normal;
  VariableCellArrayReal m_interface_distance_env;
 };

 
ARCANE_REGISTER_SUB_DOMAIN_FACTORY(HyodaEnvs, IHyodaPlugin, HyodaEnvs);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif // _HYODA_PLUGIN_ENVIRONMENTS_H_
