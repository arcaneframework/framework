// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
// ****************************************************************************
// * HyodaMix.cc                                                (C) 2000-2013 *
// ****************************************************************************
#include "arcane/IMesh.h"
#include "arcane/IApplication.h"
#include "arcane/IParallelMng.h"
#include "arcane/IVariableMng.h"
#include "arcane/ISubDomain.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/IVariableAccessor.h"
#include "arcane/ServiceBuilder.h"
#include "arcane/SharedVariable.h"
#include "arcane/MeshVariable.h"
#include "arcane/VariableRefArray.h"
#include "arcane/VariableTypes.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/AbstractService.h"
#include "arcane/VariableCollection.h"
#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/utils/IOnlineDebuggerService.h"
#include "arcane/ITransferValuesParallelOperation.h"

#include "arcane/IVariableAccessor.h"
#include "arcane/datatype/ArrayVariant.h"

#include <arcane/hyoda/Hyoda.h>
#include <arcane/hyoda/HyodaArc.h>
#include <arcane/hyoda/HyodaIceT.h>
#include <arcane/hyoda/HyodaMix.h>
#include <arcane/hyoda/IHyodaPlugin.h>

#include <IceTGL.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


// ****************************************************************************
// * INLINE STATICS OpenGL stuffs
// ****************************************************************************
inline static Real3 Perp(Real3 v){
  return Real3(v.y,-v.x,0.0);
}

static inline void glVertex(Real3 p){
  glVertex2d(p.x,p.y);
}

inline static Real3 iColor(Int32 i){
  if (i==0) return Real3(1.0, 0.0, 0.0);
  if (i==1) return Real3(1.0, 1.0, 0.0);
  if (i==2) return Real3(0.0, 1.0, 1.0);
  if (i==3) return Real3(0.0, 0.0, 1.0);
  return Real3(0.0, 0.0, 0.0);
}

inline static int glBorders(Real3 p0, Real3 p1, Real3 p2, Real3 p3, Real3 rgb){
  glColor3d(rgb.x, rgb.y, rgb.z);
  glBegin(GL_LINE_LOOP);
  glVertex(p0);
  glVertex(p1);
  glVertex(p2);
  glVertex(p3);
  glEnd();
  return 0;
}

inline static int glTri(Real3 p0, Real3 p1, Real3 p2, Real3 rgb){
  glBegin(GL_LINES);
  glColor3d(1.0,1.0,1.0);
  glVertex(p0);
  glVertex(p1);
  glColor3d(rgb.x, rgb.y, rgb.z);
  glVertex(p1);
  glVertex(p2);
  glVertex(p2);
  glVertex(p0);
  glEnd();
  glBegin(GL_POLYGON);
  glVertex(p0);
  glVertex(p1);
  glVertex(p2);
  glEnd();
  return 0;
}

inline static int glQuad(Real3 p0, Real3 p1, Real3 p2, Real3 p3, Real3 rgb){
  glBegin(GL_LINES);
  glColor3d(1.0,1.0,1.0);
  glVertex(p0);
  glVertex(p1);
  glColor3d(rgb.x, rgb.y, rgb.z);
  glVertex(p1);
  glVertex(p2);
  glVertex(p2);
  glVertex(p3);
  glVertex(p3);
  glVertex(p0);
  glEnd();
  glBegin(GL_POLYGON);
  glVertex(p0);
  glVertex(p1);
  glVertex(p2);
  glVertex(p3);
  glEnd();
  return 0;
}
inline static int glQuadFull(Real3 p0, Real3 p1, Real3 p2, Real3 p3, Real3 rgb){
  glColor3d(rgb.x, rgb.y, rgb.z);
  glBegin(GL_LINES);
  glVertex(p0);
  glVertex(p1);
  glVertex(p1);
  glVertex(p2);
  glVertex(p2);
  glVertex(p3);
  glVertex(p3);
  glVertex(p0);
  glEnd();
  glBegin(GL_POLYGON);
  glVertex(p0);
  glVertex(p1);
  glVertex(p2);
  glVertex(p3);
  glEnd();
  return 0;
}

inline static int glPenta(Real3 p0, Real3 p1, Real3 p2, Real3 p3, Real3 p4, Real3 rgb){
  glBegin(GL_LINES);
  glColor3d(1.0,1.0,1.0);
  glVertex(p0);
  glVertex(p1);
  glColor3d(rgb.x, rgb.y, rgb.z);
  glVertex(p1);
  glVertex(p2);
  glVertex(p2);
  glVertex(p3);
  glVertex(p3);
  glVertex(p4);
  glVertex(p4);
  glVertex(p0);
  glEnd();
  glBegin(GL_POLYGON);
  glVertex(p0);
  glVertex(p1);
  glVertex(p2);
  glVertex(p3);
  glVertex(p4);
  glEnd();
  return 0;
}

inline static int glHexa(Real3 p0, Real3 p1, Real3 p2, Real3 p3, Real3 p4, Real3 p5, Real3 rgb){
  glBegin(GL_LINES);
  glColor3d(1.0,1.0,1.0);
  glVertex(p0);
  glVertex(p1);
  glColor3d(rgb.x, rgb.y, rgb.z);
  glVertex(p1);
  glVertex(p2);
  glVertex(p2);
  glVertex(p3);
  glVertex(p3);
  glVertex(p4);
  glVertex(p4);
  glVertex(p5);
  glVertex(p5);
  glVertex(p0);
  glEnd();
  glBegin(GL_POLYGON);
  glVertex(p0);
  glVertex(p1);
  glVertex(p2);
  glVertex(p3);
  glVertex(p4);
  glVertex(p5);
  glEnd();
  return 0;
}

inline static int glInterface(Real3 p0, Real3 p1){
  glColor3d(1.0f, 1.0f, 1.0f);
  glBegin(GL_LINES);
  glVertex(p0);
  glVertex(p1);
  glEnd();
  return 0;
}


// ****************************************************************************
// * HyodaMix
// * InterfaceNormal
// * InterfaceDistance2 materiaux
// * InterfaceDistance2_env milieux
// ****************************************************************************
HyodaMix::HyodaMix(Hyoda *hd,
                   ISubDomain *subDomain,
                   ITraceMng *tm):TraceAccessor(tm),
                                  m_hyoda(hd),
                                  m_hPlgMats(NULL),
                                  m_hPlgEnvs(NULL),
                                  m_sub_domain(subDomain),
                                  m_default_mesh(subDomain->defaultMesh()),
                                  m_interface_normal(VariableBuildInfo(m_default_mesh, "InterfaceNormal")),
                                  m_interface_distance(VariableBuildInfo(m_default_mesh, "InterfaceDistance2ForEnv",
                                                                         IVariable::PNoDump|IVariable::PNoRestore)),
                                  coords(m_default_mesh->nodesCoordinates()),
                                  m_i_origine(VariableBuildInfo(m_default_mesh, "OriginOffset")),
                                  m_x_codes(VariableBuildInfo(m_default_mesh, "IntersectionCodes"))

{
  ServiceBuilder<IHyodaPlugin> serviceBuilder(m_sub_domain);
  m_hPlgMats = serviceBuilder.createInstance("HyodaMats", SB_AllowNull);
  m_hPlgEnvs = serviceBuilder.createInstance("HyodaEnvs", SB_AllowNull);
  if (m_hPlgMats){
    debug() << "\33[7m[HyodaMix::HyodaMix] Hyoda materials plugin loaded\33[m";
    m_hPlgMats->setHyodaMix(m_hyoda,this);
  }else{
    debug() << "\33[7m[HyodaMix::HyodaMix] NULL Hyoda materials plugin\33[m";
  }
  if (m_hPlgEnvs){
    debug() << "\33[7m[HyodaMix::HyodaMix] Hyoda environments plugin loaded\33[m";
    m_hPlgEnvs->setHyodaMix(m_hyoda,this);
  }else{
    debug() << "\33[7m[HyodaMix::HyodaMix] NULL Hyoda environments plugin\33[m";
  }
  // Les points d'intersection sont ceux possibles par coté
  // Il semble qu'on ait trois interfaces possibles au maximum
  m_p.resize(4);
  m_x.resize(12);
} 


// ****************************************************************************
// * getCellOrigin & Outils
// ****************************************************************************
typedef struct Real_Int32{
  Real dot;
  Int32 idx;
}Real_Int32;
static inline int comparOrigins(const void *one, const void *two){
  const Real_Int32 *frst=(Real_Int32*)one;
  const Real_Int32 *scnd=(Real_Int32*)two;
  if (frst->dot == scnd->dot) return 0;
  if (frst->dot <  scnd->dot) return -1;
  return +1;
}
void HyodaMix::setCellOrigin(Cell cell){
  struct Real_Int32 rtn[cell->nbNode()];
  // on profite qu'on cherche la nouvelle origine pour flusher le tableau des coordonnées des intersections
  Real3 flush(0.,0.,0.);
  m_p.fill(flush);
  m_x.fill(flush);
  //debug()<<"\t[getCellOrigin] normal="<<m_interface_normal[cell];
  for(Int32 i=0,iMx=cell->nbNode();i<iMx;i+=1){
    //debug()<<"\t[getCellOrigin] node#"<<i<<":"<< m_default_mesh->nodesCoordinates()[cell->node(i)];
    rtn[i].idx=i;
    rtn[i].dot=math::scaMul(m_interface_normal[cell], coords[cell->node(i)]);
  }
  qsort(rtn,cell->nbNode(),sizeof(Real_Int32),comparOrigins);
  // On sauve l'origine de cette maille
  m_i_origine[cell]=rtn[0].idx;
  //return m_i_origine[cell];
}


// ****************************************************************************
// * xNrmDstSgmt2Point
// * Input: l'"origine", le vecteur normal, une distance et le segment
// * Output: màj du point d'intersection s'il existe et 0||1 pour l'indiquer
// ****************************************************************************
int HyodaMix::xNrmDstSgmt2Point(Real3 p0, Real3 d0,  Real3 p1, Real3 p2, Real3 &xPoint){
  Real3 d1=p2-p1;
  Real3 delta=p1-p0;
  Real d0pd1=math::scaMul(d0,Perp(d1));
  debug()<<"\t\t[xNrmDstSgmt2Point] p0"<<p0<<", d0"<<d0;
  debug()<<"\t\t[xNrmDstSgmt2Point] Segment "<<p1<<p2;//<<", delta="<<delta<<", d0pd1="<<d0pd1;
  if (d0pd1==0.0){
    if (math::scaMul(delta,d0)==0.0){
      debug()<<"\t\t[xNrmDstSgmt2Point] The're the same";
    }else{
      debug()<<"\t\t[xNrmDstSgmt2Point] The lines are nonintersecting and parallel";
    }
    return 0;
  }
  Real inv_d0pd1=1.0/d0pd1;
  //Real s=Dot(delta,Perp(d1))*inv_d0pd1;
  Real t=math::scaMul(delta,Perp(d0))*inv_d0pd1;
  //debug()<<"\t\t[xNrmDstSgmt2Point] s="<<s;
  //debug()<<"\t\t[xNrmDstSgmt2Point] t="<<t;
  //debug()<<"\t\t[xNrmDstSgmt2Point] d1.abs()="<<d1.abs();
  if ((t>=0.0) && (t<=1.0)){
    debug()<<"\t\t[xNrmDstSgmt2Point] T Intersection here!";
    xPoint=p1+t*d1;
    return 1;   
  }  
  debug()<<"\t\t[xNrmDstSgmt2Point] No intersection here!";
  return 0; 
}


// *****************************************************************************
// * xCellPoints
// * Input: l'"origine", le vecteur normal, une distance et le segment
// * Output: màj du point d'intersection s'il existe et 0||1 pour l'indiquer
// ****************************************************************************
Int32 HyodaMix::xCellPoints(Cell c,
                            Real3 normale,
                            Real distance,
                            Int32 order){
  Int32 iOrg=m_i_origine[c];
  Int32 xCode=0;
  Real3 d0=Perp(normale);
  Real3 p0=distance*normale;  // Le vecteur d0 porte la droite
  m_p[0]=coords[c->node(iOrg)];
  m_p[1]=coords[c->node((iOrg+1)%4)];
  m_p[2]=coords[c->node((iOrg+2)%4)];
  m_p[3]=coords[c->node((iOrg+3)%4)];
  if (order>2) fatal()<<"[HyodaMix::xCellPoints] order>2";
  if (xNrmDstSgmt2Point(p0,d0,m_p[1],m_p[0],m_x[4*order+iOrg])==1) xCode|=1;
  if (xNrmDstSgmt2Point(p0,d0,m_p[2],m_p[1],m_x[4*order+((iOrg+1)%4)])==1) xCode|=2;
  if (xNrmDstSgmt2Point(p0,d0,m_p[3],m_p[2],m_x[4*order+((iOrg+2)%4)])==1) xCode|=4;
  if (xNrmDstSgmt2Point(p0,d0,m_p[0],m_p[3],m_x[4*order+((iOrg+3)%4)])==1) xCode|=8;
  // On sauve les codes
  m_x_codes[c]|=xCode<<(order<<2);
  return xCode;
}


// *****************************************************************************
// * xCellDrawInterface
// *****************************************************************************
void HyodaMix::xCellDrawInterface(Cell c, Int32 order){
  Int32 iOrg=m_i_origine[c];
  Int32 xCode=(m_x_codes[c]>>(order<<2))&0xFul;
  //debug()<<"HyodaMix::xCellDrawInterface xCode="<<xCode;  
  if (order>2) fatal()<<"xCellDrawInterface order>2";
  // On dessine les interfaces
  if (xCode==0x0) return;
  if (xCode==0xF) return;
  if (xCode==0x3) {glInterface(m_x[4*order+iOrg],         m_x[4*order+((iOrg+1)%4)]); return;}
  if (xCode==0x5) {glInterface(m_x[4*order+iOrg],         m_x[4*order+((iOrg+2)%4)]); return;}
  if (xCode==0x6) {glInterface(m_x[4*order+((iOrg+1)%4)], m_x[4*order+((iOrg+2)%4)]); return;}
  if (xCode==0x9) {glInterface(m_x[4*order+iOrg],         m_x[4*order+((iOrg+3)%4)]); return;}
  if (xCode==0xA) {glInterface(m_x[4*order+((iOrg+1)%4)], m_x[4*order+((iOrg+3)%4)]); return;}
  if (xCode==0xC) {glInterface(m_x[4*order+((iOrg+2)%4)], m_x[4*order+((iOrg+3)%4)]); return;}
  fatal()<<"HyodaMix::xCellDrawInterface Unknown! (xCode="<<xCode<<")";  
}


// *****************************************************************************
// * xCellBorders
// * hypothèses de dessin: xCodes(order)!=0 car testé avant
// *****************************************************************************
int HyodaMix::xCellBorders(Cell cell,
                           Real min, Real max,
                           Real val){
  Real3 rgb;
  // On récupère la couleur selon val et par rapport à min & max
  m_hyoda->meshIceT()->setColor(min,max,val,rgb);
  // On dessine les contours
  glBorders(coords[cell->node(0)], coords[cell->node(1)],
            coords[cell->node(2)], coords[cell->node(3)], rgb);
  return 0;
}


// *****************************************************************************
// * xCellFill
// * hypothèses de dessin: xCodes(order)!=0 car testé avant
// *****************************************************************************
int HyodaMix::xCellFill(Cell cell,
                        Int32Array& xCodes,
                        Real min, Real max,
                        Real val,
                        Int32 order,
                        Int32 nbMilieux){
  Int32 iOrg=m_i_origine[cell];
  // On va chercher la couleur de notre valeure
  Real3 rgb;
  Real3 p[4];
  Real3 x[12];
  
  debug() << "\t\t[HyodaMix::xCellFill] #"<<cell.uniqueId()
          << ": nbMilieux=" << nbMilieux
          << ", order=" << order << ", xCodes=" << xCodes;

  if (val<min) warning()<<"[HyodaMix::xCellFill] val<min " << val <<" < " << min;
  if (val>max) warning()<<"[HyodaMix::xCellFill] val>max " << val <<" > " << max;
  
  // On récupère la couleur selon val et par rapport à min & max
  m_hyoda->meshIceT()->setColor(min,max,val,rgb);
  m_hyoda->meshIceT()->checkOglError();

  //if (cell.uniqueId()==49) rgb=Real3(1.0,1.0,1.0);

  // On récupère les points dans l'ordre
  p[0]=m_p[0];
  p[1]=m_p[1];
  p[2]=m_p[2];
  p[3]=m_p[3];

  // On récupère les intersections dans l'ordre
  x[0] = m_x[4*0+iOrg];
  x[1] = m_x[4*0+((iOrg+1)%4)];
  x[2] = m_x[4*0+((iOrg+2)%4)];
  x[3] = m_x[4*0+((iOrg+3)%4)];
  x[4] = m_x[4*1+iOrg];
  x[5] = m_x[4*1+((iOrg+1)%4)];
  x[6] = m_x[4*1+((iOrg+2)%4)];
  x[7] = m_x[4*1+((iOrg+3)%4)];
  x[8] = m_x[4*2+iOrg];
  x[9] = m_x[4*2+((iOrg+1)%4)];
  x[10]= m_x[4*2+((iOrg+2)%4)];
  x[11]= m_x[4*2+((iOrg+3)%4)];
  
  if (order>2) fatal()<<"[HyodaMix::xCellFill] order>2";
  
  // Cas d'un seul milieu
  if ((nbMilieux==1 && order==0)
      || (nbMilieux==1 && xCodes[0]==0xF))
    return glQuadFull(coords[cell->node(0)], coords[cell->node(1)],
                      coords[cell->node(2)], coords[cell->node(3)], rgb);
  
  // Cas de deux milieux
  if (nbMilieux==2 && order==0) return xCellFill_i2_o0(cell,p,x,xCodes,rgb);
  if (nbMilieux==2 && order==1) return xCellFill_i2_o1(cell,p,x,xCodes,rgb);
  
  // Cas de trois milieux
  if (nbMilieux==3 && order==0) return xCellFill_i3_o0(cell,p,x,xCodes,rgb);
  if (nbMilieux==3 && order==1) return xCellFill_i3_o1(cell,p,x,xCodes,rgb);
  if (nbMilieux==3 && order==2) return xCellFill_i3_o2(cell,p,x,xCodes,rgb);
  
  fatal() << "[HyodaMix::xCellFill] Unknown"
          << ", nbMilieux=" << nbMilieux
          << ", order=" << order
          << ", with codes=" << xCodes;

  // CC : pour respecter le prototype
  return -1;
}



// *****************************************************************************
// * xCellFill_i2_o0 - Cas de 2 milieux et d'ordre 0
// *****************************************************************************
int HyodaMix::xCellFill_i2_o0(Cell c, Real3 p[4], Real3 x[12], Int32Array& xCodes, Real3 rgb){
  Int32 xCode0=xCodes.at(0);
  Int32 xCode1=xCodes.at(1);
  
  debug() << "\33[7m[xCellFill_i2_o0] xCode0="<<xCode0 <<", xCode1="<<xCode1<<"\33[m";

  // Si les deux sont vides, c'est pas normale
  if (xCode0==0x0 && xCode1==0x0)
    warning() << "HyodaMix: #" << c.uniqueId()
              << ": Case of 2 empty medium!";
  
  // Je m'impose par défaut
  if (xCode0==0xF ||(xCode0==0x0 && xCode1==0x0))
    return glQuadFull(coords[c->node(0)],
                      coords[c->node(1)],
                      coords[c->node(2)],
                      coords[c->node(3)], rgb);
  
  // Si je suis vide, je laisse la place
  if (xCode0==0x0) return 0;
  
  // Si le prochain est vide, on remplit l'espace
  //if (xCode1==0x0) return glQuadFull(coords[c->node(0)],coords[c->node(1)],coords[c->node(2)],coords[c->node(3)], rgb);
  
  // Sinon, on prend notre partie
  if (xCode0==0x3) return glPenta(x[0],x[1],p[2],p[3],p[0],rgb);
  if (xCode0==0x5) return glQuad(x[0],x[2],p[3],p[0],rgb);
  if (xCode0==0x6) return glPenta(x[1],x[2],p[3],p[0],p[1],rgb);
  if (xCode0==0x9) return glTri(x[0],x[3],p[0],rgb);
  if (xCode0==0xA) return glQuad(x[1],x[3],p[0],p[1],rgb);
  if (xCode0==0xC) return glPenta(x[2],x[3],p[0],p[1],p[2],rgb);
  warning()<<"HyodaMix::xCellFill_i2_o0 #"<<c.uniqueId()
          << "Unknown!";  
  return 0;
}


// *****************************************************************************
// * xCellFill_i2_o1 - Cas de 2 milieux et d'ordre 1
// *****************************************************************************
int HyodaMix::xCellFill_i2_o1(Cell c, Real3 p[4], Real3 x[12], Int32Array& xCodes, Real3 rgb){
  Int32 xCode0=xCodes.at(0);
  Int32 xCode1=xCodes.at(1);
  
  debug() << "\33[7m[xCellFill_i2_o1] xCode0="<<xCode0 <<", xCode1="<<xCode1<<"\33[m";

  // Si les deux sont pleins, on laisse
  if (xCode0==0xF) return 0;
  
  // Si 0 vide mais pas moi, je prends
  if (xCode0==0x0 && xCode1!=0x0) return glQuadFull(coords[c->node(0)],
                                                    coords[c->node(1)],
                                                    coords[c->node(2)],
                                                    coords[c->node(3)],rgb);

  //if (xCode0==0x3 && xCode1==0x3) return glQuad(x[0],m_x[c][4+iOrg],m_x[c][4+((iOrg+1)%4)],x[1],rgb);
  //if (xCode0==0x3 && xCode1==0xF) return glTri(x[0],p[1],x[1],rgb);
  if (xCode0==0x3) return glTri(x[1],x[0],p[1],rgb);
  
  //if (xCode0==0x5 && xCode1==0x5) return glQuad(x[0],m_x[c][4+iOrg],x[6],x[2],rgb);
  //if (xCode0==0x5 && xCode1==0x3) return glPenta(x[0],m_x[c][4+iOrg],m_x[c][4+((iOrg+1)%4)],p[2],x[2],rgb);
  //if (xCode0==0x5 && xCode1==0x6) return glPenta(x[0],p[1],m_x[c][4+((iOrg+1)%4)],x[6],x[2],rgb);
  //if (xCode0==0x5 && xCode1==0xF) return glQuad(x[0],p[1],p[2],x[2],rgb);
  if (xCode0==0x5) return glQuad(x[2],x[0],p[1],p[2],rgb);

  //if (xCode0==0x6 && xCode1==0x6) return glQuad(x[1],m_x[c][4+((iOrg+1)%4)],x[6],x[2],rgb);
  //if (xCode0==0x6 && xCode1==0xF) return glTri(x[1],p[2],x[2],rgb);
  if (xCode0==0x6) return glTri(x[2],x[1],p[2],rgb);

  //if (xCode0==0x9 && xCode1==0x3) return glHexa(x[0],m_x[c][4+iOrg],m_x[c][4+((iOrg+1)%4)],p[2],p[3],x[3],rgb);
  //if (xCode0==0x9 && xCode1==0x5) return glPenta(x[0],m_x[c][4+iOrg],x[6],p[3],x[3],rgb);
  //if (xCode0==0x9 && xCode1==0x6) return glHexa(x[0],p[1],m_x[c][4+((iOrg+1)%4)],x[6],p[3],x[3],rgb);
  //if (xCode0==0x9 && xCode1==0x9) return glQuad(x[0],m_x[c][4+iOrg],x[7],x[3],rgb);
  //if (xCode0==0x9 && xCode1==0xA) return glPenta(x[0],p[1],m_x[c][4+((iOrg+1)%4)],x[7],x[3],rgb);
  //if (xCode0==0x9 && xCode1==0xC) return glHexa(x[0],p[1],p[2],x[6],x[7],x[3],rgb);
  //if (xCode0==0x9 && xCode1==0xF) return glPenta(x[0],p[1],p[2],p[3],x[3],rgb);
  if (xCode0==0x9) return glPenta(x[3],x[0],p[1],p[2],p[3],rgb);
  
  //if (xCode0==0xA && xCode1==0xA) return glQuad(x[1],m_x[c][4+((iOrg+1)%4)],x[7],x[3],rgb);
  //if (xCode0==0xA && xCode1==0x6) return glPenta(x[1],m_x[c][4+((iOrg+1)%4)],x[6],p[3],x[3],rgb);
  //if (xCode0==0xA && xCode1==0xC) return glPenta(x[1],p[2],x[6],x[7],x[3],rgb);
  //if (xCode0==0xA && xCode1==0xF) return glQuad(x[1],p[2],p[3],x[3],rgb);
  if (xCode0==0xA) return glQuad(x[3],x[1],p[2],p[3],rgb);
  
  //if (xCode0==0xC && xCode1==0xC) return glQuad(x[6],x[2],x[3],x[7],rgb);
  //if (xCode0==0xC && xCode1==0xF) return glPenta(x[3],p[0],p[1],p[2],x[2],rgb);
  if (xCode0==0xC) return glPenta(x[2],x[3],p[0],p[1],p[2],rgb);

  warning()<<"HyodaMix::xCellFill_i2_o1 #"<<c.uniqueId()
           << ": Unknown!";  
  return 0;
}


// *****************************************************************************
// * xCellFill_i3_o0 - Cas de 3 milieux et d'ordre 0
// *****************************************************************************
int HyodaMix::xCellFill_i3_o0(Cell c, Real3 p[4], Real3 x[12], Int32Array& xCodes, Real3 rgb){
  Int32 x0=xCodes.at(0);
  Int32 x1=xCodes.at(1);
  Int32 x2=xCodes.at(2);

  //debug()<<"\tHyodaMix::xCellFill_i3_o0 " << xCodes;
  
  // Si les 3 sont vides, c'est pas normal
  if (x0==0x0 && x1==0x0 && x2==0x0)
    warning()<<"xCellFill_i3_o0 #"<<c.uniqueId()
             << ": 0x0 && 0x0!";
  
  // Je m'impose par défaut
  if (x0==0xF) return glQuadFull(coords[c->node(0)],
                                 coords[c->node(1)],
                                 coords[c->node(2)],
                                 coords[c->node(3)],rgb);
  
  // Si je suis vide, je laisse la place
  if (x0==0x0) return 0;
  
  // Si les prochains sont vides, on remplit l'espace
  if (x1==0x0 && x2==0x0) return glQuadFull(coords[c->node(0)],
                                            coords[c->node(1)],
                                            coords[c->node(2)],
                                            coords[c->node(3)],rgb);
  
  // Sinon, on prend notre partie
  if (x0==0x3) return glPenta(x[0],x[1],p[2],p[3],p[0],rgb);
  if (x0==0x5) return glQuad(x[0],x[2],p[3],p[0],rgb);
  if (x0==0x6) return glPenta(x[1],x[2],p[3],p[0],p[1],rgb);
  if (x0==0x9) return glTri(x[0],x[3],p[0],rgb);
  if (x0==0xA) return glQuad(x[1],x[3],p[0],p[1],rgb);
  if (x0==0xC) return glPenta(x[2],x[3],p[0],p[1],p[2],rgb);
  warning()<<"HyodaMix::xCellFill_i3_o0 #"<<c.uniqueId()
           << "Unknown!";  
  return 0;
}


// *****************************************************************************
// * xCellFill_i3_o1 - Cas de 3 milieux et d'ordre 1
// *****************************************************************************
int HyodaMix::xCellFill_i3_o1(Cell c, Real3 p[4], Real3 x[12], Int32Array& xCodes, Real3 rgb){
  Int32 x0=xCodes.at(0);
  Int32 x1=xCodes.at(1);
  Int32 x2=xCodes.at(2);
  
  //debug()<<"\tHyodaMix::xCellFill_i3_o1 " << xCodes;

  // Si les 3 sont pleins, on laisse
  if (x0==0xF && x1==0xF && x2==0xF) return 0;
  
  // Si 0&1 vide mais pas moi, je prends
  if (x0==0x0 && x1==0x0) return glQuadFull(coords[c->node(0)],
                                            coords[c->node(1)],
                                            coords[c->node(2)],
                                            coords[c->node(3)],rgb);
  
  if (x0==0x0 && x1==0x5) return glQuad(x[4],x[6],p[3],p[0],rgb);
  if (x0==0x0 && x1==0x6) return glPenta(x[5],x[6],p[3],p[0],p[1],rgb);
  if (x0==0x0 && x1==0xA) return glQuad(x[5],x[7],p[0],p[1],rgb);

  if (x0==0x3 && x1==0x3) return glQuad(x[1],x[0],x[4],x[5],rgb);
  if (x0==0x3 && x1==0xF) return glTri(x[1],x[0],p[1],rgb);
  
  if (x0==0x5 && x1==0x5) return glQuad(x[4],x[6],x[2],x[0],rgb);
  if (x0==0x5 && x1==0x3) return glPenta(x[4],x[5],p[2],x[2],x[0],rgb);
  if (x0==0x5 && x1==0x6) return glPenta(x[5],x[6],x[2],x[0],p[1],rgb);
  if (x0==0x5 && x1==0xF) return glQuad(x[2],x[0],p[1],p[2],rgb);

  if (x0==0x6 && x1==0x6) return glQuad(x[5],x[6],x[2],x[1],rgb);
  if (x0==0x6 && x1==0xF) return glTri(x[2],x[1],p[2],rgb);

  if (x0==0x9 && x1==0x3) return glHexa(x[4],x[5],p[2],p[3],x[3],x[0],rgb);
  if (x0==0x9 && x1==0x5) return glPenta(x[4],x[6],p[3],x[3],x[0],rgb);
  if (x0==0x9 && x1==0x6) return glHexa(x[5],x[6],p[3],x[3],x[0],p[1],rgb);
  if (x0==0x9 && x1==0x9) return glQuad(x[4],x[7],x[3],x[0],rgb);
  if (x0==0x9 && x1==0xA) return glPenta(x[5],x[7],x[3],x[0],p[1],rgb);
  if (x0==0x9 && x1==0xC) return glHexa(x[6],x[7],x[3],x[0],p[1],p[2],rgb);
  if (x0==0x9 && x1==0xF) return glPenta(x[3],x[0],p[1],p[2],p[3],rgb);
  
  if (x0==0xA && x1==0xA) return glQuad(x[5],x[7],x[3],x[1],rgb);
  if (x0==0xA && x1==0x6) return glPenta(x[5],x[6],p[3],x[3],x[1],rgb);
  if (x0==0xA && x1==0xC) return glPenta(x[6],x[7],x[3],x[1],p[2],rgb);
  if (x0==0xA && x1==0xF) return glQuad(x[3],x[1],p[2],p[3],rgb);
  
  if (x0==0xC && x1==0xC) return glQuad(x[7],x[6],x[2],x[3],rgb);
  if (x0==0xC && x1==0xF) return glPenta(x[2],x[3],p[0],p[1],p[2],rgb);

  warning() << "\33[7m[HyodaEnvs::xCellFill_i3_o1] #"<<c.uniqueId()
            << ": unhandled " << xCodes <<"\33[m";
  return 0;
}


// *****************************************************************************
// * xCellFill_i3_o2 - Cas de 3 milieux et d'ordre 2
// *****************************************************************************
int HyodaMix::xCellFill_i3_o2(Cell c, Real3 p[4], Real3 x[12], Int32Array& xCodes, Real3 rgb){
  Int32 x0=xCodes.at(0);
  Int32 x1=xCodes.at(1);
  Int32 x2=xCodes.at(2);

  //debug()<<"\tHyodaMix::xCellFill_i3_o2 " << xCodes;

  // 0||1 ont déjà tout pris
  if (x0==0xF || x1==0xF) return 0;
  
  // 2 premiers vides et pas moi = je prends tout
  if (x0==0x0 && x1==0x0 && x2!=0x0) return glQuadFull(coords[c->node(0)],
                                                       coords[c->node(1)],
                                                       coords[c->node(2)],
                                                       coords[c->node(3)],rgb);

  if (x0==0x0 && x1==0x5 && x2==0x6) return glPenta(x[9],x[10],x[6],x[4],p[1],rgb);
  if (x0==0x0 && x1==0x6 && x2==0x6) return glQuad(x[9],x[10],x[6],x[5],rgb);
  if (x0==0x0 && x1==0x6 && x2==0xF) return glTri(x[6],x[5],p[2],rgb);
  if (x0==0x0 && x1==0xA && x2==0xF) return glQuad(x[7],x[5],p[2],p[3],rgb);

  if (x0==0x5 && x1==0x5 && x2==0x5) return glQuad(x[8],x[10],x[6],x[4],rgb);
  if (x0==0x5 && x1==0x5 && x2==0x6) return glPenta(x[9],x[10],x[6],x[4],p[1],rgb);
  if (x0==0x5 && x1==0x5 && x2==0xF) return glQuad(x[6],x[4],p[1],p[2],rgb);
  if (x0==0x5 && x1==0x6 && x2==0x6) return glQuad(x[9],x[10],x[6],x[5],rgb);
  if (x0==0x5 && x1==0x6 && x2==0xF) return glTri(x[6],x[5],p[2],rgb);
  
  if (x0==0x6 && x1==0x6 && x2==0x6) return glQuad(x[9],x[10],x[6],x[5],rgb);
  if (x0==0x6 && x1==0x6 && x2==0xF) return glTri(x[6],x[5],p[2],rgb);
  
  if (x0==0x9 && x1==0x5 && x2==0x5) return glQuad(x[8],x[10],x[6],x[4],rgb);
  if (x0==0x9 && x1==0x5 && x2==0xF) return glQuad(x[6],x[4],p[1],p[2],rgb);
  if (x0==0x9 && x1==0x5 && x2==0x6) return glPenta(x[9],x[10],x[6],x[4],p[1],rgb);
  if (x0==0x9 && x1==0x6 && x2==0x6) return glQuad(x[9],x[10],x[6],x[5],rgb);
  if (x0==0x9 && x1==0x6 && x2==0xF) return glTri(x[6],x[5],p[2],rgb);
  if (x0==0x9 && x1==0x9 && x2==0x6) return glHexa(x[9],x[10],p[3],x[7],x[4],p[1],rgb);
  if (x0==0x9 && x1==0x9 && x2==0xF) return glPenta(x[7],x[4],p[1],p[2],p[3],rgb);
  if (x0==0x9 && x1==0xA && x2==0x6) return glPenta(x[9],x[10],p[3],x[7],x[5],rgb);
  if (x0==0x9 && x1==0xA && x2==0xF) return glQuad(x[7],x[5],p[2],p[3],rgb);

  if (x0==0xA && x1==0x6 && x2==0x6) return glQuad(x[9],x[10],x[6],x[5],rgb);
  if (x0==0xA && x1==0x6 && x2==0xF) return glTri(x[6],x[5],p[2],rgb);
  if (x0==0xA && x1==0xA && x2==0x6) return glPenta(x[9],x[10],p[3],x[7],x[5],rgb);
  if (x0==0xA && x1==0xA && x2==0xA) return glQuad(x[9],x[11],x[7],x[5],rgb);
  if (x0==0xA && x1==0xA && x2==0xF) return glQuad(x[7],x[5],p[2],p[3],rgb);
  if (x0==0xA && x1==0xC && x2==0xF) return glPenta(x[6],x[7],p[0],p[1],p[2],rgb);
  warning() << "\33[7m[HyodaEnvs::xCellFill_i3_o2] #"<< c.uniqueId()
            << ": unhandled " << xCodes <<"\33[m";
  return 0;
}


// *****************************************************************************
// * xLine2Cell
// * On devrait vérifier que la distance considérée est bien >0.0 et < diagonale
// * depuis l'"origine"
// ****************************************************************************
void HyodaMix::xLine2Cell(int plugin, IVariable *variable, Real min, Real max){
  // Si QHyoda réclame les globals, c'est pas normal ici
  if (plugin==0)
    fatal() << "\33[7m[HyodaEnvs::xLine2Cell] plugin==0\33[m";

  // Si QHyoda réclame les 'environments', on les lui affiche depuis le plugin HyodaEnvs
  if (plugin==1){
    if (m_hPlgEnvs!=NULL){
      if (variable->itemKind()!=IK_Cell)
        throw FatalErrorException("[HyodaMix::xLine2Cell] QHyoda 'env' plugin only support IK_Cell!");
      m_hPlgEnvs->draw(variable,min,max);
    }else
      debug()<<"\t\33[7m[xLine2Cell] plugin==2 & Null m_hPlgEnvs!\33[m";
  }

  // Si QHyoda réclame les 'materials', on les lui affiche depuis le plugin HyodaMats
  if (plugin==2){
    if (m_hPlgMats!=NULL)
      m_hPlgMats->draw(variable,min,max);
    else
      debug()<<"\t\33[7m[xLine2Cell] plugin==1 & Null m_hPlgMats!\33[m";
  }


}


// *****************************************************************************
// * xCellDrawNormal
// *****************************************************************************
void HyodaMix::
xCellDrawNormal(Cell c, Real3 p[4], Int32 iDst)
{
  ARCANE_UNUSED(iDst);
  if (m_interface_normal[c].abs()==0.0)
    return;
  // Maille en cours
  glBegin(GL_POINTS);
  glColor3d(1.0, 1.0, 1.0); glVertex(p[0]);
  glColor3d(1.0, 0.0, 0.0); glVertex(p[1]);
  glColor3d(0.0, 1.0, 0.0); glVertex(p[2]);
  glColor3d(0.0, 0.0, 1.0); glVertex(p[3]);
  glEnd();
  // Vecteur org à org+normale
  glColor3d(1.0, 1.0, 1.0);
  glBegin(GL_LINES);
  glVertex(p[0]);
  glVertex(p[0]+m_interface_normal[c]);
  glEnd();
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
