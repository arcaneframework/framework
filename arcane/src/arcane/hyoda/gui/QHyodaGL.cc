// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <QtWidgets>
#include <QHyodaGL.h>

#ifndef GL_MULTISAMPLE
#define GL_MULTISAMPLE  0x809D
#endif

QHyodaGL::QHyodaGL(QWidget *parent):QOpenGLWidget(parent){
  qDebug()<<"\33[31m[QHyodaGL::QHyodaGL] NEW\33[m";
  pov.setX(30.0);
  pov.setY(-45.0);
  pov.setZ(0.0);
  pov.setW(100.0);
}

void QHyodaGL::initializeGL(void){
  qDebug()<<"\33[31m[QHyodaGL::initializeGL]\33[m";
  glPointSize(8.0);
  glLineWidth(1.0);
  glShadeModel(GL_SMOOTH);
  glEnable(GL_MULTISAMPLE);
  glEnable(GL_POINT_SMOOTH);
  glClearDepth(1.0f);
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LEQUAL);
  //glEnable(GL_CULL_FACE);
  glLineStipple(1, 0x0101); // Mode pointillÃ©
  glEnable(GL_LINE_STIPPLE);
}

void QHyodaGL::paintGL(){
  //qDebug()<<"\33[31m[QHyodaGL::paintGL]\33[m";
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glPushMatrix();
  glLoadIdentity();
  glTranslatef(0.0f, 0.0f, -16.0f);
  glRotatef(pov.x(), 1.0f, 0.0f, 0.0f);
  glRotatef(pov.y(), 0.0f, 1.0f, 0.0f);
  glRotatef(pov.z(), 0.0f, 0.0f, 1.0f);
  glScalef(pov.w(), pov.w(), pov.w());
  draw();
  glPopMatrix();
}

void QHyodaGL::resizeGL(int w, int h){
  //qDebug()<<"\33[31m[QHyodaGL::resizeGL]\33[m";
  const int side = qMin(w, h);
  glViewport((w-side)/2,(h-side)/2,side,side);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(-2.0f,+2.0f,-2.0f,+2.0f,1.0f,32.0f);
  glMatrixMode(GL_MODELVIEW);
}

void QHyodaGL::mousePressEvent(QMouseEvent *event){
  //qDebug()<<"\33[31m[QHyodaGL::mousePressEvent]\33[m";
  mouse = QVector2D(event->localPos());
}

void QHyodaGL::mouseMoveEvent(QMouseEvent *event){
  //qDebug()<<"\33[31m[QHyodaGL::mouseMoveEvent]\33[m";
  QVector2D diff = QVector2D(event->localPos())-mouse;
  if (event->buttons() & Qt::LeftButton) {
    pov.setX(pov.x()+diff.y()/4.0f);
    pov.setY(pov.y()+diff.x()/4.0f);
  } else if (event->buttons() & Qt::RightButton) {
    pov.setX(pov.x()+diff.y()/4.0f);
    pov.setZ(pov.z()+diff.x()/4.0f);
  }
  mouse = QVector2D(event->localPos());
}

void QHyodaGL::wheelEvent(QWheelEvent *e){
  //qDebug()<<"\33[31m[QHyodaGL::mouseWheelEvent]\33[m";
  e->delta()>0?
    pov.setW(pov.w()+pov.w()*0.1f):
    pov.setW(pov.w()-pov.w()*0.1f);
  //update();
}
