%module(directors="1") ArcaneLauncher

%import core/ArcaneSwigCore.i

%{
#include "ArcaneSwigUtils.h"
#include "arcane/launcher/ArcaneLauncher.h"
using namespace Arcane;
%}

#define ARCANE_LAUNCHER_EXPORT

%ignore Arcane::ArcaneLauncher::runDirect;
%ignore Arcane::ArcaneLauncher::setCommandLineArguments;
%ignore Arcane::ArcaneLauncher::createStandaloneAcceleratorMng;
%rename Arcane::ArcaneLauncher ArcaneLauncher_INTERNAL;

ARCANE_STD_EXHANDLER
%include arcane/launcher/ArcaneLauncher.h
%exception;
