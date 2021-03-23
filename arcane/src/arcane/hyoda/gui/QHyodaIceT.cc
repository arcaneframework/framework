// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <QtWidgets>
#include <QHyodaIceT.h>

/******************************************************************************
 * QHyodaIceT class
 *****************************************************************************/
QHyodaIceT::QHyodaIceT(QWidget *prnt): QOpenGLWidget(prnt),
                                       parent(prnt),
                                       mouse(QVector2D()),
                                       pov(QVector4D(13.25,26.75,0,0.73)),
                                       image(NULL),
                                       texture(0){
  qDebug()<<"\33[31m[QHyodaIceT::QHyodaIceT] NEW\33[m";
}



/******************************************************************************
 * ~QHyodaIceT
 *****************************************************************************/
QHyodaIceT::~QHyodaIceT(){
  qDebug()<<"~QHyodaIceT";
}


// ****************************************************************************
// * sxyz
// * tcpMeshIceTImage utilise pour rÃ©cupÃ©rer le pov
// ****************************************************************************
void QHyodaIceT::sxyz(double *v){
  v[0]=pov.w();
  v[1]=pov.x();
  v[2]=pov.y();
  v[3]=pov.z();
}

void QHyodaIceT::setPov(QVector4D v){
  pov=v;
}


/******************************************************************************
 * setImage
 *****************************************************************************/
void QHyodaIceT::setImage(QImage *qImg){
  //qDebug()<<"\33[31m[QHyodaIceT::setImage]\33[m";
  image=qImg;
  update();
}


/******************************************************************************
 * initializeGL
 *****************************************************************************/
void QHyodaIceT::initializeGL(){
  qDebug()<<"\33[31m[QHyodaIceT::initializeGL]\33[m";
  glViewport(0, 0, parent->width(), parent->height());
  glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_NORMALIZE);
  glPixelStorei(GL_UNPACK_ALIGNMENT,1);  
  glEnable(GL_TEXTURE_2D);
  glGenTextures(1,&texture);
  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  //glBindTexture(GL_TEXTURE_2D, 0);
}


/******************************************************************************
 * resizeGL
 *****************************************************************************/
void QHyodaIceT::resizeGL(int w, int h){
  //qDebug()<<"\33[31m[QHyodaIceT::resizeGL]\33[m";
  glViewport(0.0, 0.0, w, h);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glFrustum(-1.0f, 1.0f, -1.0f, 1.0f, +1, 8);
  glTranslatef(0.0f, 0.0f, -2.0f);
  update();
}


/******************************************************************************
 * paintGL
 *****************************************************************************/
void QHyodaIceT::
paintGL()
{
  if (!image)
    return;
  //qDebug()<<"\33[31m[QHyodaIceT::paintGL] "<<image->width()<<"x"<<image->height()<<"\33[m";
  const double sqrtd=2.0;//sqrt(3.0);
  saveGLState();
  
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
               image->width(), image->height(), 0,
               GL_RGBA, GL_UNSIGNED_BYTE, image->bits());
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  
  glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);  
  glBegin(GL_QUADS);
  glTexCoord2d(0.0, 0.0); glVertex3d(-sqrtd, -sqrtd, 0.0);
  glTexCoord2d(0.0, 1.0); glVertex3d(-sqrtd, +sqrtd, 0.0);
  glTexCoord2d(1.0, 1.0); glVertex3d(+sqrtd, +sqrtd, 0.0);
  glTexCoord2d(1.0, 0.0); glVertex3d(+sqrtd, -sqrtd, 0.0);
  glEnd();

  restoreGLState();
}


/******************************************************************************
 * mousePressEvent
 *****************************************************************************/
void QHyodaIceT::mousePressEvent(QMouseEvent *e){
  //qDebug()<<"\33[31m[QHyodaIceT::mousePressEvent]\33[m";
  mouse = QVector2D(e->localPos());
}

/******************************************************************************
 * mouseMoveEvent
 *****************************************************************************/
void QHyodaIceT::mouseMoveEvent(QMouseEvent *e){
  //qDebug()<<"\33[31m[QHyodaIceT::mouseMoveEvent]\33[m";
  QVector2D diff = QVector2D(e->localPos())-mouse;

  if (e->buttons() & Qt::LeftButton) {
    pov.setX(pov.x()+diff.y()/4.0f);
    pov.setY(pov.y()+diff.x()/4.0f);
  } else if (e->buttons() & Qt::RightButton) {
    pov.setX(pov.x()+diff.y()/4.0f);
    pov.setZ(pov.z()+diff.x()/4.0f);
  }
  mouse = QVector2D(e->localPos());
  update();
}

/******************************************************************************
 * wheelEvent
 *****************************************************************************/
void QHyodaIceT::wheelEvent(QWheelEvent *e){
  //qDebug()<<"\33[31m[QHyodaIceT::wheelEvent]\33[m";
  e->delta()>0?
    pov.setW(pov.w()+pov.w()*0.1f):
    pov.setW(pov.w()-pov.w()*0.1f);
  update(); 
}


/******************************************************************************
 * saveGLState
 *****************************************************************************/
void QHyodaIceT::saveGLState(){
  glPushAttrib(GL_ALL_ATTRIB_BITS);
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
}


/******************************************************************************
 * restoreGLState
 *****************************************************************************/
void QHyodaIceT::restoreGLState(){
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
  glPopAttrib();
}
