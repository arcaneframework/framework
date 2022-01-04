// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
// ****************************************************************************
// * HyodaMatrix.cc                                             (C) 2000-2016 *
// ****************************************************************************
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

#ifndef GL_MULTISAMPLE
#define GL_MULTISAMPLE  0x809D
#endif

#include "arcane/hyoda/HyodaIceTGlobal.h"
#include "arcane/hyoda/Hyoda.h"
#include "arcane/hyoda/HyodaArc.h"
#include "arcane/hyoda/HyodaTcp.h"
#include "arcane/hyoda/HyodaIceT.h"
#include "arcane/hyoda/HyodaMatrix.h"
#include "arcane/hyoda/HyodaMix.h"
extern void ppmWrite(const char*, const unsigned char*, int width, int height);

namespace
{
  static Arcane::HyodaMatrix* HyodaMatrixCallbackHandle = NULL;
  void drawGLCallback(void){
    if (!HyodaMatrixCallbackHandle) return;
    HyodaMatrixCallbackHandle->drawGL();
  }
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


/******************************************************************************
 * HyodaMatrix
 *****************************************************************************/
HyodaMatrix::HyodaMatrix(Hyoda *hd,
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
                                          scale(0.),
                                          rot_x(0.),
                                          rot_y(0.),
                                          rot_z(0.),
                                          m_tcp(_tcp),
                                          m_hyoda_matrix_cpu(0)
                                          //m_hyoda_matrix_n(0),
                                          //m_hyoda_matrix_i(NULL),
                                          //m_hyoda_matrix_j(NULL),
                                          //m_hyoda_matrix_val(NULL)
{
  debug()<<"\33[35m[HyodaMatrix::HyodaMatrix] New "
        << ", screen_width="<<m_screen_width
        << ", screen_height="<<m_screen_height
        << "\33[m";

  m_hyoda_matrix_n=(int*)calloc(8,sizeof(int));
  m_hyoda_matrix_i=(int**)calloc(8,sizeof(int*));
  m_hyoda_matrix_j=(int**)calloc(8,sizeof(int*));
  m_hyoda_matrix_val=(double**)calloc(8,sizeof(double*));
  
  m_pov_sxyzip[0]=
    m_pov_sxyzip[1]=
     m_pov_sxyzip[2]=
      m_pov_sxyzip[3]=
       m_pov_sxyzip[4]=
        m_pov_sxyzip[5]=0.0;

  /* specify Z, stencil, accum sizes */
  osMesaContext = OSMesaCreateContextExt(OSMESA_RGBA, 8, 0, 0, NULL);
  if (!osMesaContext) fatal()<<"\33[35mOSMesaCreateContext failed!\33[m\n";
  /* Allocate the image buffer */
  m_image_buffer=malloc(renderSize());
  if (!m_image_buffer) fatal()<<"\33[35mAlloc image failed!\33[m\n";
  /* Bind the image to the context and make it current */
  if (!OSMesaMakeCurrent(osMesaContext,
                         m_image_buffer,
                         GL_UNSIGNED_BYTE,
                         m_screen_width,
                         m_screen_height ))
    fatal()<<"\33[35mOSMesaMakeCurrent failed!\33[m\n";
  {
    int z, s, a;
    glGetIntegerv(GL_DEPTH_BITS, &z);
    glGetIntegerv(GL_STENCIL_BITS, &s);
    glGetIntegerv(GL_ACCUM_RED_BITS, &a);
    debug()<<"\33[35m[HyodaMatrix] Depth="<<z<<", Stencil="<<s<<", Accum="<<a<<"\33[m";
  }
  // IceT system init
 #warning IceT needs MPI_Init
  //IceTBitField diag_level = ICET_DIAG_ALL_NODES | ICET_DIAG_WARNINGS;
  if (sd->parallelMng()->isParallel()){
    debug() << "\33[35m[HyodaMatrix] isParallel icetCreateMPICommunicator @"
           << sd->parallelMng()->getMPICommunicator()<< "\33[m";
    icetCommunicator = icetCreateMPICommunicator(*(MPI_Comm*)sd->parallelMng()->getMPICommunicator());
  }else{
    debug()<<"\33[35m[HyodaMatrix] MPI_COMM_SELF icetCreateMPICommunicator\33[m";
    icetCommunicator = icetCreateMPICommunicator(MPI_COMM_SELF);
  }
  icetContext = icetCreateContext(icetCommunicator);
  
  // iceT GL
  icetGLInitialize();
  icetGLDrawCallback(drawGLCallback);
  HyodaMatrixCallbackHandle=this;
  icetResetTiles();
  icetAddTile(0, 0, m_screen_width, m_screen_height, 0);
  icetStrategy(ICET_STRATEGY_SEQUENTIAL);
  icetSingleImageStrategy(ICET_SINGLE_IMAGE_STRATEGY_AUTOMATIC);
  //icetDiagnostics(diag_level);
  icetSetColorFormat(ICET_IMAGE_COLOR_RGBA_UBYTE);
  icetSetDepthFormat(ICET_IMAGE_DEPTH_FLOAT);
  icetCompositeMode(ICET_COMPOSITE_MODE_Z_BUFFER);
  icetBoundingBoxd(-1.0f, +1.0f, -1.0f, +1.0f, -4.0f, 16.0f);
  setLeftRightBottomTop();
  icetDisable(ICET_FLOATING_VIEWPORT);
  icetEnable(ICET_ORDERED_COMPOSITE);
  initGL();
  checkIceTError();
  debug()<<"\33[35m[HyodaMatrix] checkOglError\33[m";
  checkOglError();
  traceMng()->flush();
 }


// ****************************************************************************
// * setIJVal
// ****************************************************************************
void HyodaMatrix::setIJVal(int cpu, int n, int *i, int *j, double *val){
  m_hyoda_matrix_cpu=math::max(cpu,m_hyoda_matrix_cpu);
  m_hyoda_matrix_n[cpu]=n;
  m_hyoda_matrix_i[cpu]=i;
  m_hyoda_matrix_j[cpu]=j;
  m_hyoda_matrix_val[cpu]=val;
  debug()<<"\33[7m[HyodaMatrix::setIJVal] cpu="<<cpu<<",n="<<n
        <<", m_hyoda_matrix_cpu="<<m_hyoda_matrix_cpu<<"\33[m";
}


/******************************************************************************
 * Initialisation de la base ortho qui servira à glOrtho: left, right, bottom, top
 *****************************************************************************/
void HyodaMatrix::setLeftRightBottomTop(){
  lrbtnf[0]=lrbtnf[1]=lrbtnf[2]=lrbtnf[3]=0.0;
  lrbtnf[4]=-4.0; // On commence avec un Z-min par défaut pour le 2D
  lrbtnf[5]=+16.0; // Idem avec le Z-max, pour le 2D par défaut

  int min_i=+123456789;
  int max_i=-123456789;
  int min_j=+123456789;
  int max_j=-123456789;
  
  for(int cpu=0;cpu<=m_hyoda_matrix_cpu;cpu+=1){
    debug()<<"\33[7m[HyodaMatrix::setLeftRightBottomTop] cpu="<<cpu<<"/"<<m_hyoda_matrix_cpu
          <<", m_hyoda_matrix_n[cpu]="<<m_hyoda_matrix_n[cpu]<<"\33[m";
    for(int k=0;k<m_hyoda_matrix_n[cpu];k+=1){
      min_i = math::min(min_i,m_hyoda_matrix_i[cpu][k]);
      max_i = math::max(max_i,m_hyoda_matrix_i[cpu][k]);
      min_j = math::min(min_j,m_hyoda_matrix_j[cpu][k]);
      max_j = math::max(max_j,m_hyoda_matrix_j[cpu][k]);
    }
  }
  lrbtnf[0]= min_j; // left
  lrbtnf[1]= max_j; // right
  lrbtnf[2]= min_i; // bottom
  lrbtnf[3]= max_j; // top
  debug()<<"\33[35m[HyodaMatrix::setLeftRightBottomTop] min_i="<< min_i<<"\33[m";
  debug()<<"\33[35m[HyodaMatrix::setLeftRightBottomTop] max_i="<< max_i<<"\33[m";
  debug()<<"\33[35m[HyodaMatrix::setLeftRightBottomTop] min_j="<< min_j<<"\33[m";
  debug()<<"\33[35m[HyodaMatrix::setLeftRightBottomTop] max_j="<< max_j<<"\33[m";

  UniqueArray<Real> rdcMin(3);
  UniqueArray<Real> rdcMax(3);
  rdcMin[0]=lrbtnf[0]; // left
  rdcMin[1]=lrbtnf[2]; // bottom
  rdcMin[2]=lrbtnf[4]; // near
  m_sub_domain->parallelMng()->reduce(Parallel::ReduceMin,rdcMin.view());
  rdcMax[0]=lrbtnf[1]; // right
  rdcMax[1]=lrbtnf[3]; // top
  rdcMax[2]=lrbtnf[5]; // far
  m_sub_domain->parallelMng()->reduce(Parallel::ReduceMax,rdcMax.view());
  lrbtnf[0]=rdcMin[0]; // left
  lrbtnf[1]=rdcMax[0]; // right
  lrbtnf[2]=rdcMin[1]; // bottom
  lrbtnf[3]=rdcMax[1]; // top
  lrbtnf[4]=rdcMin[2]; // near
  lrbtnf[5]=rdcMax[2]; // far
  debug()<<"\33[35m[HyodaMatrix::setLeftRightBottomTop] left ="   << lrbtnf[0]<<"\33[m";
  debug()<<"\33[35m[HyodaMatrix::setLeftRightBottomTop] right ="  << lrbtnf[1]<<"\33[m";
  debug()<<"\33[35m[HyodaMatrix::setLeftRightBottomTop] bottom =" << lrbtnf[2]<<"\33[m";
  debug()<<"\33[35m[HyodaMatrix::setLeftRightBottomTop] top ="    << lrbtnf[3]<<"\33[m";
  debug()<<"\33[35m[HyodaMatrix::setLeftRightBottomTop] near ="   << lrbtnf[4]<<"\33[m";
  debug()<<"\33[35m[HyodaMatrix::setLeftRightBottomTop] far ="    << lrbtnf[5]<<"\33[m";
}


/******************************************************************************
 * sxyz
 *****************************************************************************/
void HyodaMatrix::sxyzip(double *pov){
  scale=pov[0];
  rot_x=pov[1];
  rot_y=pov[2];
  rot_z=pov[3];
  debug()<<"\33[35m[HyodaMatrix::sxyzi] scale="<<scale<<
    ", rot_x="<<rot_x<<", rot_y="<<rot_y<<", rot_z="<<rot_z<<"\33[m";
}


/******************************************************************************
 * imgSize
 *****************************************************************************/
int HyodaMatrix::renderSize(void){
  return m_screen_width*m_screen_height*4*sizeof(GLubyte);
}


/******************************************************************************
 * ~HyodaMatrix
 *****************************************************************************/
HyodaMatrix::~HyodaMatrix(){
  free(m_image_buffer);
  checkOglError();
  checkIceTError();
  debug()<<"\33[35m~HyodaMatrix"<<"\33[m";
  icetDestroyContext(icetContext);
  OSMesaDestroyContext(osMesaContext);
}



/******************************************************************************
 * render
 *****************************************************************************/
void HyodaMatrix::render(void){    
  debug()<<"\33[35m[HyodaMatrix] render\33[m";
  icetGLDrawCallback(drawGLCallback);
  // Tout le monde participe au rendering
  // Mais on blinde cette section contre les erreurs FPU qu'iceT a tendance à jeter
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
    
    // Si la variable d'environement ARCANE_HYODA_MATRIX_PPM est settée, on dump l'image dans un fichier
    if (!platform::getEnvironmentVariable("ARCANE_HYODA_MATRIX_PPM").null()){
      char filename[3883];
      snprintf(filename,3883,"/tmp/HyODA_Mtx_%04d.ppm",
               m_sub_domain->commonVariables().globalIteration());
      debug()<<"\33[33m[HyodaIceT::render] \33[33mARCANE_HYODA_MATRIX_PPM\33[m "
             << icetImageGetWidth(image) << "x" << icetImageGetHeight(image)
             << " image " << filename;
      ppmWrite(filename, imgAddress,(int)m_screen_width, (int)m_screen_height);
    }

    debug()<<"\33[35m[HyodaMatrix::render] sending header packet for image!\33[m";
    char icePacketHeader[8];
    // On écrit le 'MatrixIceTHeader' QHyodaTcpSwitch
    *(unsigned int*)&icePacketHeader[0]=0x78f78f67ul;
    // On pousse la taille du paquet
    *(unsigned int*)&icePacketHeader[4]=8;
    m_tcp->send(icePacketHeader,8);
    m_tcp->waitForAcknowledgment();
    debug()<<"\33[35m[HyodaMatrix::render] sending image packet!, renderSize="<<renderSize()<<"\33[m";
    m_tcp->send((void*)imgAddress, renderSize());
    m_tcp->waitForAcknowledgment();
    debug()<<"\33[35m[HyodaMatrix::render] Waiting for POV answer!\33[m";
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
  debug()<<"\33[35m[Hyoda::hook] broadcasting pov..."<<"\33[m";
  m_sub_domain->parallelMng()->broadcast(pov_to_broadcast.view(),0);
  m_pov_sxyzip[0]=pov_to_broadcast[0];
  m_pov_sxyzip[1]=pov_to_broadcast[1];
  m_pov_sxyzip[2]=pov_to_broadcast[2];
  m_pov_sxyzip[3]=pov_to_broadcast[3];
  m_pov_sxyzip[4]=pov_to_broadcast[4];
  m_pov_sxyzip[5]=pov_to_broadcast[5];
  sxyzip(m_pov_sxyzip);
  debug()<<"\33[35m[Hyoda::hook] pov done"<<"\33[m";  
  debug()<<"\33[35m[HyodaMatrix::render] sxyzip:"
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
void HyodaMatrix::initGL(void){
  debug()<<"\33[35m[HyodaMatrix] initGL in"<<"\33[m";
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
  debug()<<"\33[35m[HyodaMatrix] initGL out"<<"\33[m";
  checkOglError();
}


/******************************************************************************
 * drawGLCallback
 *****************************************************************************/
void HyodaMatrix::drawGL(void){
  debug()<<"\33[35m[HyodaMatrix] drawGL"<<"\33[m";
  glPushMatrix();
  glShadeModel(GL_SMOOTH);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  setLeftRightBottomTop();
    
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
  drawMatrix();
  glPopMatrix();
  glFlush();
  glFinish();
  checkOglError();
}

// ****************************************************************************
// * OpenGL tools
// ****************************************************************************
static inline void glVertex(Real3 p){
  glVertex2d(p.x,p.y);
}

inline static int glQuadFull(Real3 p0, Real3 p1, Real3 p2, Real3 p3, Real3 rgb){
  glColor3d(rgb.x, rgb.y, rgb.z);
  //glBegin(GL_LINES);
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

/******************************************************************************
 * drawArcaneMesh
 *****************************************************************************/
void HyodaMatrix::drawMatrix(void){
  debug()<<"\33[35m[HyodaMatrix] drawMatrix"<<"\33[m";
  Real3 rgb;
  Real imin=+HUGE_VAL; Real imax=-HUGE_VAL;
  Real jmin=+HUGE_VAL; Real jmax=-HUGE_VAL;
  Real cmin=+HUGE_VAL; Real cmax=-HUGE_VAL;
  int n=0;
  for(int cpu=0;cpu<=m_hyoda_matrix_cpu;cpu+=1){
    iceRowMinMax(cpu,imin,imax);
    iceColMinMax(cpu,jmin,jmax);
    iceValMinMax(cpu,cmin,cmax);
    n+=m_hyoda_matrix_n[cpu];
  }
  debug()<<"\33[35m[HyodaMatrix] imin="<<imin<<", imax="<<imax<<"\33[m";
  debug()<<"\33[35m[HyodaMatrix] jmin="<<jmin<<", jmax="<<jmax<<"\33[m";
  debug()<<"\33[35m[HyodaMatrix] cmin="<<cmin<<", cmax="<<cmax<<"\33[m";
  debug()<<"\33[35m[HyodaMatrix] n="<<n<<"\33[m";
  for(int cpu=0;cpu<=m_hyoda_matrix_cpu;cpu+=1){
    for(int k=0;k<m_hyoda_matrix_n[cpu];k+=1){
      const double value=m_hyoda_matrix_j[cpu][k];
      const Real3 p(imax-m_hyoda_matrix_i[cpu][k],m_hyoda_matrix_j[cpu][k],0);
      const Real3 n0=p+Real3(-0.5,-0.5,0.0);
      const Real3 n1=p+Real3(+0.5,-0.5,0.0);
      const Real3 n2=p+Real3(+0.5,+0.5,0.0);
      const Real3 n3=p+Real3(-0.5,+0.5,0.0);
      glBegin(GL_LINES);
      setColor(cmin,cmax,value,rgb); 
      glQuadFull(n0,n1,n2,n3,rgb);
    }
    checkOglError();
  }
  checkOglError();
}


// ****************************************************************************
// * iceColorMinMax
// ****************************************************************************
void HyodaMatrix::iceValMinMax(int cpu,Real &min, Real &max){
  for(int k=0;k<m_hyoda_matrix_n[cpu];k+=1){
    const Real val=m_hyoda_matrix_j[cpu][k];
    min = math::min(min,val);
    max = math::max(max,val);
  }
  //min=m_sub_domain->parallelMng()->reduce(Parallel::ReduceMin, min);
  //max=m_sub_domain->parallelMng()->reduce(Parallel::ReduceMax, max);
  debug()<<"\33[35m[iceValMinMax] cpu="<<cpu<<", min="<<min<<", max="<<max<<"\33[m";
}


void HyodaMatrix::iceRowMinMax(int cpu, Real &min, Real &max){
  for(int k=0;k<m_hyoda_matrix_n[cpu];k+=1){
    const Real val=m_hyoda_matrix_i[cpu][k];
    min = math::min(min,val);
    max = math::max(max,val);
  }
  //min=m_sub_domain->parallelMng()->reduce(Parallel::ReduceMin, min);
  //max=m_sub_domain->parallelMng()->reduce(Parallel::ReduceMax, max);
  debug()<<"\33[35m[iceRowMinMax] cpu="<<cpu<<", min="<<min<<", max="<<max<<"\33[m";
}

void HyodaMatrix::iceColMinMax(int cpu, Real &min, Real &max){
  for(int k=0;k<m_hyoda_matrix_n[cpu];k+=1){
    const Real val=m_hyoda_matrix_j[cpu][k];
    min = math::min(min,val);
    max = math::max(max,val);
  }
  //min=m_sub_domain->parallelMng()->reduce(Parallel::ReduceMin, min);
  //max=m_sub_domain->parallelMng()->reduce(Parallel::ReduceMax, max);
  debug()<<"\33[35m[iceColMinMax] cpu="<<cpu<<", min="<<min<<", max="<<max<<"\33[m";
}



/******************************************************************************
 * color
 *****************************************************************************/
void HyodaMatrix::
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
void HyodaMatrix::checkIceTError(void){
#define CASE_ERROR(ename) case ename: debug() << "## IceT status = \33[35m" << #ename << "\33[m"; break;
  switch (icetGetError()) {
  case(ICET_NO_ERROR): break;
  CASE_ERROR(ICET_SANITY_CHECK_FAIL);
  CASE_ERROR(ICET_INVALID_ENUM);
  CASE_ERROR(ICET_BAD_CAST);
  CASE_ERROR(ICET_OUT_OF_MEMORY);
  CASE_ERROR(ICET_INVALID_OPERATION);
  CASE_ERROR(ICET_INVALID_VALUE);
  default:debug()<<"\33[35m## UNKNOWN ICET ERROR CODE!!!!!\33[m";
  }
#undef CASE_ERROR
}


/******************************************************************************
 * checkMatrixError
 *****************************************************************************/
void HyodaMatrix::checkOglError(void){
#define CASE_ERROR(ename) case ename: debug() << "\33[35m## OpenGL status = " << #ename << "\33[m"; break;
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
  default:debug()<<"\33[35m## UNKNOWN GL ERROR CODE!!!!!\33[m";
  }
#undef CASE_ERROR
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


