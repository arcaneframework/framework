// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*****************************************************************************
 * HyodaIceT.cc                                                (C) 2000-2012 *
 *****************************************************************************/

#include "arcane/IMesh.h"
#include "arcane/IApplication.h"
#include "arcane/IParallelMng.h"
#include "arcane/IVariableMng.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/IVariableAccessor.h"
#include "arcane/FactoryService.h"
#include "arcane/ServiceFinder2.h"
#include "arcane/SharedVariable.h"
#include "arcane/CommonVariables.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/AbstractService.h"
#include "arcane/VariableCollection.h"
#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/IOnlineDebuggerService.h"
#include "arcane/ITransferValuesParallelOperation.h"

#include "arcane/IVariableAccessor.h"
#include <arcane/IParticleFamily.h>
#include "arcane/datatype/ArrayVariant.h"

#ifndef GL_MULTISAMPLE
#define GL_MULTISAMPLE  0x809D
#endif

#include "arcane/hyoda/HyodaIceTGlobal.h"
#include "arcane/hyoda/Hyoda.h"
#include "arcane/hyoda/HyodaArc.h"
#include "arcane/hyoda/HyodaTcp.h"
#include "arcane/hyoda/HyodaIceT.h"
#include "arcane/hyoda/HyodaMix.h"
extern void ppmWrite(const char*, const unsigned char*, int width, int height);

namespace
{
  static Arcane::HyodaIceT* HyodaIceTCallbackHandle = NULL;
  void drawGLCallback(void){
    if (!HyodaIceTCallbackHandle) return;
    HyodaIceTCallbackHandle->drawGL();
  }
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


/******************************************************************************
 * HyodaIceT
 *****************************************************************************/
HyodaIceT::HyodaIceT(Hyoda *hd,
                     ISubDomain *sd,
                     ITraceMng *tm,
                     unsigned int w,
                     unsigned int h,
                     HyodaTcp *_tcp): TraceAccessor(tm),
                                      hyoda(hd),
                                      m_sub_domain(sd),
                                      icetContext(NULL),
                                      icetCommunicator(NULL),
                                      osMesaContext(NULL),
                                      rank(sd->parallelMng()->commRank()),
                                      num_proc(sd->parallelMng()->commSize()),
                                      m_screen_width(w),
                                      m_screen_height(h),
                                      m_image_buffer(NULL),
                                      scale(1.),
                                      rot_x(0.1),
                                      rot_y(0.),
                                      rot_z(0.),
                                      variable_index(0),
                                      variable_plugin(0),
                                      m_tcp(_tcp),
                                      m_hyoda_mix(new HyodaMix(hd,sd,tm)),
                                      m_variable(NULL)
{
  debug()<<"\33[33m[HyodaIceT::HyodaIceT] New "
        << ", screen_width="<<m_screen_width
        << ", screen_height="<<m_screen_height
        << "\33[m";
  
  m_pov_sxyzip[0]=
    m_pov_sxyzip[1]=
     m_pov_sxyzip[2]=
      m_pov_sxyzip[3]=
       m_pov_sxyzip[4]=
        m_pov_sxyzip[5]=0.0;

  #warning Duplicated code here
  // On cherche la premier IVariable à afficher
  VariableCollection variables = m_sub_domain->variableMng()->usedVariables();
  for(VariableCollection::Enumerator ivar(variables); ++ivar; ){
    IVariable* var = *ivar;
    // Pas de références, pas de variable
    if (var->nbReference()==0) {debug() << "\33[33m[HyodaIceT::HyodaIceT] No nbReference\33[m"; continue;}
    // Pas sur le bon support, pas de variable
    if (var->itemKind() != IK_Node &&
        var->itemKind() != IK_Cell &&
        var->itemKind() != IK_Face &&
        var->itemKind() != IK_Particle) continue;
    // Pas réclamée en tant que PostProcess'able, pas de variable
    if (var->itemKind() != IK_Particle && (!var->hasTag("PostProcessing")))
      {debug() << "\33[33m[HyodaIceT::HyodaIceT] "<<var->name()<<" not PostProcessing\33[m"; continue;}
    // Pas de type non supportés
    //if (var->dataType()>=DT_String) continue;
    if (var->dataType()!=DT_Real) continue;
    m_variable=var;
    debug() << "\33[33m[HyodaIceT::HyodaIceT] m_variable is "<<var->name()<<"\33[m";
    break;
  }
  if (!m_variable)
    ARCANE_FATAL("\33[33m[HyodaIceT::HyodaIceT] No m_variable to draw!\33[m");
  debug()<<"\33[33m[HyodaIceT::HyodaIceT] Focusing on variable: "<<m_variable->name()<<"\33[m";

  /* specify Z, stencil, accum sizes */
  osMesaContext = OSMesaCreateContextExt(OSMESA_RGBA, 8, 0, 0, NULL);
  if (!osMesaContext) fatal()<<"OSMesaCreateContext failed!\n";
  /* Allocate the image buffer */
  m_image_buffer=malloc(renderSize());
  if (!m_image_buffer) fatal()<<"Alloc image failed!\n";
  /* Bind the image to the context and make it current */
  if (!OSMesaMakeCurrent(osMesaContext,
                         m_image_buffer,
                         GL_UNSIGNED_BYTE,
                         m_screen_width,
                         m_screen_height ))
    fatal()<<"OSMesaMakeCurrent failed!\n";
  {
    int z, s, a;
    glGetIntegerv(GL_DEPTH_BITS, &z);
    glGetIntegerv(GL_STENCIL_BITS, &s);
    glGetIntegerv(GL_ACCUM_RED_BITS, &a);
    debug()<<"\33[33m[HyodaIceT] Depth="<<z<<", Stencil="<<s<<", Accum="<<a<<"\33[m";
  }
  // IceT system init
 #warning IceT needs MPI_Init
  IceTBitField diag_level = ICET_DIAG_ALL_NODES | ICET_DIAG_WARNINGS;
  if (sd->parallelMng()->isParallel()){
    debug() << "\33[33m[HyodaIceT] isParallel icetCreateMPICommunicator @"
           << sd->parallelMng()->getMPICommunicator()<< "\33[m";
    icetCommunicator = icetCreateMPICommunicator(*(MPI_Comm*)sd->parallelMng()->getMPICommunicator());
  }else{
    debug()<<"\33[33m[HyodaIceT] MPI_COMM_SELF icetCreateMPICommunicator\33[m";
    icetCommunicator = icetCreateMPICommunicator(MPI_COMM_SELF);
  }
  icetContext = icetCreateContext(icetCommunicator);
  
  // iceT GL
  icetGLInitialize();
  icetGLDrawCallback(drawGLCallback);
  HyodaIceTCallbackHandle=this;
  icetResetTiles();
  icetAddTile(0, 0, m_screen_width, m_screen_height, 0);
  icetStrategy(ICET_STRATEGY_SEQUENTIAL);
  icetSingleImageStrategy(ICET_SINGLE_IMAGE_STRATEGY_AUTOMATIC);
  //icetDiagnostics(diag_level);
  icetSetColorFormat(ICET_IMAGE_COLOR_RGBA_UBYTE);
  icetSetDepthFormat(ICET_IMAGE_DEPTH_FLOAT);
  icetCompositeMode(ICET_COMPOSITE_MODE_Z_BUFFER);
  //icetBoundingBoxd(-1.0f, +1.0f, -1.0f, +1.0f, -4.0f, 16.0f);
  setLeftRightBottomTop();
  Real length=lrbtnf[1]-lrbtnf[0];
  Real height=lrbtnf[3]-lrbtnf[2];
  icetBoundingBoxd(lrbtnf[0]-length, lrbtnf[1]+length, lrbtnf[2]-height, lrbtnf[3]+height, lrbtnf[4], lrbtnf[5]);
  icetDisable(ICET_FLOATING_VIEWPORT);
  icetEnable(ICET_ORDERED_COMPOSITE);
  initGL();
  checkIceTError();
  debug()<<"\33[33m[HyodaIceT] checkOglError\33[m";
  checkOglError();
  traceMng()->flush();
 }



/******************************************************************************
 * Initialisation de la base ortho qui servira à glOrtho: left, right, bottom, top
 *****************************************************************************/
void HyodaIceT::setLeftRightBottomTop(void){
  Real scale=1.0;
  const VariableItemReal3 nodes_coords=m_sub_domain->defaultMesh()->nodesCoordinates();
  debug()<<"[HyodaIceT::setLeftRightBottomTop] mesh()->dimension()="<<m_sub_domain->mesh()->dimension();
  lrbtnf[0]=lrbtnf[1]=lrbtnf[2]=lrbtnf[3]=0.0;
  lrbtnf[4]=-4.0; // On commence avec un Z-min par défaut pour le 2D
  lrbtnf[5]=+16.0; // Idem avec le Z-max, pour le 2D par défaut
  //if (m_sub_domain->mesh()->dimension()==3) scale=1.0;
  ENUMERATE_NODE(node,m_sub_domain->defaultMesh()->allCells().outerFaceGroup().nodeGroup()){
    //ENUMERATE_NODE(node, m_sub_domain->defaultMesh()->ownNodes()){
    lrbtnf[0]=math::min(lrbtnf[0],nodes_coords[node].x); // left
    lrbtnf[1]=math::max(lrbtnf[1],nodes_coords[node].x); // right
    lrbtnf[2]=math::min(lrbtnf[2],nodes_coords[node].y); // bottom
    lrbtnf[3]=math::max(lrbtnf[3],nodes_coords[node].y); // top
    lrbtnf[4]=math::min(lrbtnf[4],nodes_coords[node].z); // near
    lrbtnf[5]=math::max(lrbtnf[5],nodes_coords[node].z); // far
  }
  UniqueArray<Real> rdcMin(3);
  UniqueArray<Real> rdcMax(3);
  rdcMin[0]=scale*lrbtnf[0]; // left
  rdcMin[1]=scale*lrbtnf[2]; // bottom
  rdcMin[2]=scale*lrbtnf[4]; // near
  m_sub_domain->parallelMng()->reduce(Parallel::ReduceMin,rdcMin.view());
  rdcMax[0]=scale*lrbtnf[1]; // right
  rdcMax[1]=scale*lrbtnf[3]; // top
  rdcMax[2]=scale*lrbtnf[5]; // far
  m_sub_domain->parallelMng()->reduce(Parallel::ReduceMax,rdcMax.view());
  lrbtnf[0]=rdcMin[0]; // left
  lrbtnf[1]=rdcMax[0]; // right
  lrbtnf[2]=rdcMin[1]; // bottom
  lrbtnf[3]=rdcMax[1]; // top
  lrbtnf[4]=rdcMin[2]; // near
  lrbtnf[5]=rdcMax[2]; // far
  debug()<<"\33[33m[HyodaIceT::setLeftRightBottomTop] left ="   << lrbtnf[0]<<"\33[m";
  debug()<<"\33[33m[HyodaIceT::setLeftRightBottomTop] right ="  << lrbtnf[1]<<"\33[m";
  debug()<<"\33[33m[HyodaIceT::setLeftRightBottomTop] bottom =" << lrbtnf[2]<<"\33[m";
  debug()<<"\33[33m[HyodaIceT::setLeftRightBottomTop] top ="    << lrbtnf[3]<<"\33[m";
  debug()<<"\33[33m[HyodaIceT::setLeftRightBottomTop] near ="   << lrbtnf[4]<<"\33[m";
  debug()<<"\33[33m[HyodaIceT::setLeftRightBottomTop] far ="    << lrbtnf[5]<<"\33[m";
}


/******************************************************************************
 * sxyz
 *****************************************************************************/
void HyodaIceT::sxyzip(double *pov){
  scale=pov[0];
  rot_x=pov[1];
  rot_y=pov[2];
  rot_z=pov[3];
  debug()<<"\33[33m[HyodaIceT::sxyzi] scale="<<scale<<", rot_x="<<rot_x<<", rot_y="<<rot_y<<", rot_z="<<rot_z<<"\33[m";
  variable_index=(int)pov[4];
  variable_plugin=(int)pov[5];
  if (variable_index<0){
    debug()<<"\33[33m[HyodaIceT::sxyzi] Fetched negative variable_index!\33[m";
    variable_index=0;
  }
  if (variable_plugin<0 || variable_plugin>2){
    debug()<<"\33[33m[HyodaIceT::sxyzi] Fetched uncompatible variable_plugin!\33[m";
    variable_plugin=0;
  }

  // Scrutation de la variable ARCANE ciblée pour être affichée
  int i=0;
  VariableCollection variables = m_sub_domain->variableMng()->usedVariables();
  debug()<<"\33[33m[HyodaIceT::sxyzi] Looking for variable #"<<variable_index<<"\33[m";
  for(VariableCollection::Enumerator ivar(variables); ++ivar;){
    m_variable = *ivar;
    if (m_variable->nbReference()==0) continue;
    if (m_variable->itemKind()!=IK_Node &&
        m_variable->itemKind()!=IK_Cell &&
        m_variable->itemKind()!=IK_Face &&
        m_variable->itemKind()!=IK_Particle) continue;
    if (m_variable->itemKind()!=IK_Particle && (!m_variable->hasTag("PostProcessing"))) continue;
    //if (variable->dataType()>=DT_String) continue;
    if (m_variable->dataType()!=DT_Real) continue;
    if (i==variable_index) break;
    if (i>variable_index)
      fatal()<<"\33[33m[HyodaIceT::sxyzi] Could not find variable to draw (variable_index="<<variable_index<<", i="<<i<<")!\33[m\n";
    i+=1;
  }
  debug()<<"\33[33m[HyodaIceT::sxyzi] " << m_variable->name()<<"\33[m";
}


/******************************************************************************
 * imgSize
 *****************************************************************************/
int HyodaIceT::renderSize(void){
  return m_screen_width*m_screen_height*4*sizeof(GLubyte);
}


/******************************************************************************
 * ~HyodaIceT
 *****************************************************************************/
HyodaIceT::~HyodaIceT(){
  free(m_image_buffer);
  checkOglError();
  checkIceTError();
  debug()<<"\33[33m~HyodaIceT\33[m";
  icetDestroyContext(icetContext);
  OSMesaDestroyContext(osMesaContext);
}



/******************************************************************************
 * render
 *****************************************************************************/
void HyodaIceT::render(void){    
  debug()<<"\33[33m[HyodaIceT] render\33[m";
  icetGLDrawCallback(drawGLCallback);
  // Tout le monde participe au rendering
  // Mais on blinde cette section contre les erreurs FPU qu'iceT a tendance à jeter
  //platform::enableFloatingException(false);
  platform::enableFloatingException(false);
  IceTImage image=icetGLDrawFrame();
  platform::enableFloatingException(true);
  checkOglError();
  checkIceTError();
  
  m_sub_domain->parallelMng()->barrier();

  // Seul le CPU 0 possède l'image résultante pour l'instant
  if (rank==0){
    // Now fetch image's address
    unsigned char* imgAddress=icetImageGetColorub(image);

    // Si la variable d'environement ARCANE_HYODA_MESH_PPM est settée, on dump l'image dans un fichier
    if (!platform::getEnvironmentVariable("ARCANE_HYODA_MESH_PPM").null()){
      char filename[3883];
      if (29!=snprintf(filename,3883,"/tmp/HyODA_Msh_%04d.ppm", m_sub_domain->commonVariables().globalIteration()))
        return;
      debug()<<"\33[33m[HyodaIceT::render] \33[33mARCANE_HYODA_MESH_PPM\33[m "
            << icetImageGetWidth(image) << "x" << icetImageGetHeight(image)
            << " image " << filename;
      ppmWrite(filename, imgAddress,(int)m_screen_width, (int)m_screen_height);
    }
  
    debug()<<"\33[33m[HyodaIceT::render] sending header packet for image!\33[m";
    char icePacketHeader[8];
    // On écrit le 'MeshIceTHeader' QHyodaTcpSwitch
    *(unsigned int*)&icePacketHeader[0]=0xcbce69bcul;
    // On pousse la taille du paquet
    *(unsigned int*)&icePacketHeader[4]=8;
    m_tcp->send(icePacketHeader,8);
    m_tcp->waitForAcknowledgment();
    debug()<<"\33[33m[HyodaIceT::render] sending image packet!, renderSize="<<renderSize()<<"\33[m";
    m_tcp->send((void*)imgAddress, renderSize());
    m_tcp->waitForAcknowledgment();
    debug()<<"\33[33m[HyodaIceT::render] Waiting for POV answer!\33[m";
    m_tcp->recvPov(m_pov_sxyzip);
  }
  
  m_sub_domain->parallelMng()->barrier();

  // On dépiote la réponse
  UniqueArray<Real> pov_to_broadcast(0);
  pov_to_broadcast.add(m_pov_sxyzip[0]);
  pov_to_broadcast.add(m_pov_sxyzip[1]);
  pov_to_broadcast.add(m_pov_sxyzip[2]);
  pov_to_broadcast.add(m_pov_sxyzip[3]);
  pov_to_broadcast.add(m_pov_sxyzip[4]);
  pov_to_broadcast.add(m_pov_sxyzip[5]);
  debug()<<"\33[33m[Hyoda::hook] broadcasting pov...\33[m";
  m_sub_domain->parallelMng()->broadcast(pov_to_broadcast.view(),0);
  m_pov_sxyzip[0]=pov_to_broadcast[0];
  m_pov_sxyzip[1]=pov_to_broadcast[1];
  m_pov_sxyzip[2]=pov_to_broadcast[2];
  m_pov_sxyzip[3]=pov_to_broadcast[3];
  m_pov_sxyzip[4]=pov_to_broadcast[4];
  m_pov_sxyzip[5]=pov_to_broadcast[5];
  sxyzip(m_pov_sxyzip);
  debug()<<"\33[33m[Hyoda::hook] pov done\33[m";  
  debug()<<"\33[33m[HyodaIceT::render] sxyzip:"
        << "scale=" << m_pov_sxyzip[0]
        << ", x=" << m_pov_sxyzip[1]
        << ", y=" << m_pov_sxyzip[2]
        << ", z=" << m_pov_sxyzip[3]
        << ", i=" << m_pov_sxyzip[4]
        << ", p=" << m_pov_sxyzip[5]
        << "\33[m";
}


/******************************************************************************
 * GL init
 *****************************************************************************/
void HyodaIceT::initGL(void){
  debug()<<"\33[33m[HyodaIceT] initGL in\33[m";
  checkOglError();
  glViewport(0, 0, m_screen_width,m_screen_height);
  glEnable(GL_DITHER);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_MULTISAMPLE);
  glEnable(GL_POINT_SMOOTH);
  //glEnable(GL_POLYGON_SMOOTH);
  //glEnable(GL_LINE_SMOOTH);
  glPointSize(1.0);
  glLineWidth(1.0);
  glDepthFunc(GL_LESS);
  glShadeModel(GL_SMOOTH);
  glShadeModel(GL_FLAT);
  glLineStipple(1, 0x0101);
  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  glClearDepth(1.0);
  glColor3f(0.0f, 0.0f, 0.0f);      // Black points
  glClearColor(0.0, 0.0, 0.0, 0.0); // Black background
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  debug()<<"\33[33m[HyodaIceT] initGL out\33[m";
  checkOglError();
}


/******************************************************************************
 * drawGLCallback
 *****************************************************************************/
void HyodaIceT::drawGL(void){
  debug()<<"\33[33m[HyodaIceT] drawGL\33[m";
  glPushMatrix();
  glShadeModel(GL_SMOOTH);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  Real length=lrbtnf[1]-lrbtnf[0];
  Real height=lrbtnf[3]-lrbtnf[2];
  //glOrtho(lrbtnf[0], lrbtnf[1], lrbtnf[2], lrbtnf[3], lrbtnf[4], lrbtnf[5]);
  glOrtho(lrbtnf[0]-length/4., lrbtnf[1]+length/4., lrbtnf[2]-height/4., lrbtnf[3]+height/4., lrbtnf[4], lrbtnf[5]);
  //glFrustum(lrbtnf[0], lrbtnf[1], lrbtnf[2], lrbtnf[3], lrbtnf[4], lrbtnf[5]);
  //glTranslatef(0.0f, 0.0f, -4.0f);
  // et on applique les transformation venant de l'IHM
  glRotatef(rot_x, 1.0f, 0.0f, 0.0f);
  glRotatef(rot_y, 0.0f, 1.0f, 0.0f);
  glRotatef(rot_z, 0.0f, 0.0f, 1.0f);
  glScalef(scale, scale, scale);
  glMatrixMode(GL_MODELVIEW);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  drawArcaneMesh();
  glPopMatrix();
  glFlush();
  glFinish();
  checkOglError();
}


/******************************************************************************
 * drawArcaneMesh
 *****************************************************************************/
void HyodaIceT::drawArcaneMesh(void){
  //const register VariableItemReal3 nodes_coords=m_sub_domain->defaultMesh()->nodesCoordinates();
  // S'il n'y a pas de variable à afficher, il n'y a rien à scruter
  /*if (m_tcp->getVariablesNumber()==0 || m_tcp->getVariablesNumber()==-1) {
    debug()<<"[HyodaIceT::drawArcaneMesh] No variables have been set as PostProcessing!";
    drawArcPolygonsWithoutVariable();
    return;
    }*/
  debug()<<"\33[33m[HyodaIceT] drawArcaneMesh\33[m";
  drawArcPolygons();
  checkOglError();
}


// ****************************************************************************
// * iceColorMinMax
// ****************************************************************************
void HyodaIceT::iceColorMinMax(Real &min, Real &max){
  String item_family_name(m_variable->itemFamilyName()==NULL?
                          "particles":m_variable->itemFamilyName());
  VariableItemReal var(VariableBuildInfo(m_sub_domain->defaultMesh(),
                                         m_variable->name(),item_family_name),
                       m_variable->itemKind());
  ENUMERATE_ITEM(item, m_variable->itemFamily()->allItems().own()){
    Real val=var[item];
    min = math::min(min,val);
    max = math::max(max,val);
  }
  min=m_sub_domain->parallelMng()->reduce(Parallel::ReduceMin, min);
  max=m_sub_domain->parallelMng()->reduce(Parallel::ReduceMax, max);
  debug()<<"\33[33m[iceColorMinMax] min="<<min<<", max="<<max<<"\33[m";
}


// *****************************************************************************
// * drawArcPolygons
// *****************************************************************************
void HyodaIceT::drawArcPolygons(void){
  Real3 rgb;
  Real min=+HUGE_VAL;
  Real max=-HUGE_VAL;
  String item_family_name(m_variable->itemFamilyName()==NULL?
                          "particles":m_variable->itemFamilyName());
  VariableNodeReal3 nCoords = m_sub_domain->defaultMesh()->nodesCoordinates();
  VariableItemReal var(VariableBuildInfo(m_sub_domain->defaultMesh(),
                                         m_variable->name(),item_family_name),
                       m_variable->itemKind());
  
  // On calcule le min max des variables globales
  iceColorMinMax(min,max);
  if (m_sub_domain->defaultMesh()->dimension()==2){
    debug()<<"\33[33m[drawArcPolygons] 2D\33[m";
    // On regarde si un plugins est réclamé depuis l'interface
    if (variable_plugin!=0){
      debug()<<"\33[33m[drawArcPolygons] variable_plugin="<<variable_plugin<<"\33[m";
      m_hyoda_mix->xLine2Cell(variable_plugin, m_variable, min, max);
      //debug()<<"[HyodaIceT] drawArcPolygons xLine2Cell";
      checkOglError();
      return;
    }
    
    debug()<<"\33[33m[drawArcPolygons] variable_plugin="<<variable_plugin<<"\33[m";
    if ((m_variable->itemKind()==IK_Cell)||(m_variable->itemKind()==IK_Node)){
      debug()<<"\33[33m[drawArcPolygons] m_variable->name()="<<m_variable->name()<<"\33[m";
      debug()<<"\33[33m[drawArcPolygons] m_variable->itemKind()="<<m_variable->itemKind()<<"\33[m";
      ENUMERATE_CELL(cell, m_sub_domain->defaultMesh()->ownCells()){
        glBegin(GL_POLYGON);
        if (m_variable->itemKind()==IK_Cell) setColor(min,max,var[cell],rgb); 
        ENUMERATE_NODE(node, cell->nodes()){
          if (m_variable->itemKind()==IK_Node) setColor(min,max,var[node],rgb);
          glColor3d(rgb[0], rgb[1], rgb[2]);
          glVertex2d(nCoords[node].x, nCoords[node].y);
        }
        glEnd();
      }
      //debug()<<"[HyodaIceT] drawArcPolygons CELL NODE";
      checkOglError();
      return;
    }
    
    if (m_variable->itemKind()==IK_Face){
      debug()<<"\33[33m[drawArcPolygons] IK_Face!\33[m";
      ENUMERATE_FACE(face, m_sub_domain->defaultMesh()->ownFaces()){
        glBegin(GL_LINES);
        setColor(min,max,var[face],rgb);
        glColor3d(rgb[0], rgb[1], rgb[2]);
        ENUMERATE_NODE(node, face->nodes()){
          glVertex2d(nCoords[node].x, nCoords[node].y);
        }
        glEnd();  
      }
      //debug()<<"[HyodaIceT] drawArcPolygons FACE NODE";
      checkOglError();
      return;
    }
  } // dimension()==2

  if (m_sub_domain->defaultMesh()->dimension()==3){
    debug()<<"\33[33m[drawArcPolygons] 3D\33[m";
    if (m_variable->itemKind()==IK_Cell){
      ENUMERATE_FACE(face, m_sub_domain->defaultMesh()->outerFaces()){
        if (!face->isOwn()) continue;
        glBegin(GL_POLYGON);
        setColor(min,max,var[face->cell(0)],rgb);
        glColor3d(rgb[0], rgb[1], rgb[2]);
        ENUMERATE_NODE(node, face->nodes()){
          glVertex3d(nCoords[node].x,
                     nCoords[node].y,
                     nCoords[node].z);
        }
        glEnd();
      }
    }
    
    if (m_variable->itemKind()==IK_Particle){
      VariableParticleReal3 m_particle_r(VariableBuildInfo(m_sub_domain->defaultMesh(),"r","particles"));
      ENUMERATE_PARTICLE(particle, m_sub_domain->defaultMesh()->findItemFamily("particles",true)->allItems()){
        if (!particle->cell().isOwn()) continue;
        glBegin(GL_POINTS);
        setColor(min,max,var[particle],rgb);
        glColor3d(rgb[0], rgb[1], rgb[2]);
        glVertex3d(m_particle_r[particle].x,m_particle_r[particle].y,m_particle_r[particle].z);
        glEnd();
      }
    }
    
    // Permet de dessiner les lignes des arêtes
    /*glBegin(GL_LINE_STRIP);
      glColor3d(0.0, 0.0, 0.0);
      ENUMERATE_NODE(node, m_sub_domain->defaultMesh()->outerFaces()){
      glVertex3d(nCoords[node].x,
      nCoords[node].y,
      nCoords[node].z);
      }
      glEnd();*/
  } // dimension()==3
}


// ****************************************************************************
// * drawArcPolygons
// ****************************************************************************
void HyodaIceT::drawArcPolygonsWithoutVariable(void){
  VariableNodeReal3 nCoords = m_sub_domain->defaultMesh()->nodesCoordinates();
  ENUMERATE_CELL(cell, m_sub_domain->defaultMesh()->ownCells()){
    glBegin(GL_POLYGON);
    glColor3d(0.0, 0.0, 1.0);
    ENUMERATE_NODE(node, cell->nodes()){
      glVertex3d(nCoords[node].x,
                 nCoords[node].y,
                 nCoords[node].z);
    }
    glEnd();  
  }
}


/******************************************************************************
 * drawArcPoints
 *****************************************************************************/
void HyodaIceT::drawArcPoints(const VariableItemReal3 &nodes_coords,
                              VariableItemReal &var,
                              double min,double max){
  Real3 rgb;
  // On trace les points
  glBegin(GL_POINTS);
  /*ENUMERATE_NODE(node, m_sub_domain->defaultMesh()->ownNodes()){
    setColor(min,max,var[node],rgb);
    glColor3d(rgb[0], rgb[1], rgb[2]);
    glVertex3d(nodes_coords[node].x,
               nodes_coords[node].y,
               nodes_coords[node].z);
               }*/
  ENUMERATE_CELL(cell, m_sub_domain->defaultMesh()->ownCells()){
    #warning Only cell Variables for drawArcPoints
    setColor(min,max,var[cell],rgb);
    ENUMERATE_NODE(node, cell->nodes()){
      glColor3d(rgb[0], rgb[1], rgb[2]);
      glVertex3d(nodes_coords[node].x,
                 nodes_coords[node].y,
                 nodes_coords[node].z);
    }
  }
  glEnd();
}


/******************************************************************************
 * drawArcLines
 *****************************************************************************/
void HyodaIceT::drawArcLines(const VariableItemReal3 &nodes_coords,
                             VariableItemReal&){
// On trace les lignes
  glBegin(GL_LINES);
  //glColor3f(1.0, 1.0, 1.0);
  glColor3f(0.0f, 0.0f, 0.0f);
  ENUMERATE_FACE(face, m_sub_domain->defaultMesh()->allFaces()){
    for(Integer i=0,mx=face->nodes().size()-1;i<mx;++i){
      glVertex3d(nodes_coords[face->node(i)].x,
                 nodes_coords[face->node(i)].y,
                 nodes_coords[face->node(i)].z);
      glVertex3d(nodes_coords[face->node(i+1)].x,
                 nodes_coords[face->node(i+1)].y,
                 nodes_coords[face->node(i+1)].z);
    }
    glVertex3d(nodes_coords[face->node(0)].x,
               nodes_coords[face->node(0)].y,
               nodes_coords[face->node(0)].z);
    glVertex3d(nodes_coords[face->node(face->nodes().size()-1)].x,
               nodes_coords[face->node(face->nodes().size()-1)].y,
               nodes_coords[face->node(face->nodes().size()-1)].z);
  }
  glEnd();
}


/******************************************************************************
 * color
 *****************************************************************************/
void HyodaIceT::
setColor(double min, double max, double v, Real3 &rgb)
{
  ARCANE_ASSERT(min<=max,("setColor min<=max"));
  if (min==max){
    min=0.0;
    max=1.0;
  }
  double mid=max-min;
  ARCANE_ASSERT(mid!=0.,("setColor mid!=0."));
  rgb.x=rgb.y=rgb.z=1.0;
  if (v<min) v=min;
  if (v>max) v=max;
  if (v<(min+0.25*mid)){
    rgb.x=0.;
    rgb.y=4.*(v-min)/mid;
  }else if (v<(min+0.5*mid)){
    rgb.x=0.;
    rgb.z=1.+4*(min+0.25*mid-v)/mid;
  }else if (v<(min+0.75*mid)){
    rgb.x=4*(v-min-0.5*mid)/mid;
    rgb.z=0.;
  }else{
    rgb.y=1.+4*(min+0.75*mid-v)/mid;
    rgb.z=0.;
   }
}


/******************************************************************************
 * checkIceTError
 *****************************************************************************/
void HyodaIceT::checkIceTError(void){
#define CASE_ERROR(ename) case ename: debug() << "\33[33m## IceT status = " << #ename << "\33[m"; break;
  switch (icetGetError()) {
  case(ICET_NO_ERROR): break;
  CASE_ERROR(ICET_SANITY_CHECK_FAIL);
  CASE_ERROR(ICET_INVALID_ENUM);
  CASE_ERROR(ICET_BAD_CAST);
  CASE_ERROR(ICET_OUT_OF_MEMORY);
  CASE_ERROR(ICET_INVALID_OPERATION);
  CASE_ERROR(ICET_INVALID_VALUE);
  default:debug()<<"\33[33m## UNKNOWN ICET ERROR CODE!!!!!\33[m";
  }
#undef CASE_ERROR
}


/******************************************************************************
 * checkIceTError
 *****************************************************************************/
void HyodaIceT::checkOglError(void){
#define CASE_ERROR(ename) case ename: debug() << "\33[33m## OpenGL status = " << #ename << "\33[m"; break;
  switch (glGetError()) {
  case(GL_NO_ERROR):break;
  CASE_ERROR(GL_INVALID_ENUM);
  CASE_ERROR(GL_INVALID_VALUE);
  CASE_ERROR(GL_INVALID_OPERATION);
  CASE_ERROR(GL_STACK_OVERFLOW);
  CASE_ERROR(GL_STACK_UNDERFLOW);
  CASE_ERROR(GL_OUT_OF_MEMORY);
#ifdef GL_TABLE_TOO_LARGE
  CASE_ERROR(GL_TABLE_TOO_LARGE);
#endif
  default:debug()<<"\33[33m## UNKNOWN GL ERROR CODE!!!!!\33[m";
  }
#undef CASE_ERROR
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

