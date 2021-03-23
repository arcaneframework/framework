//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Text.RegularExpressions;
using System.Xml.Linq;

namespace Arcane.Curves
{
  /// <summary>
  /// Classe permettant de renommer les courbes d'un cas.
  /// </summary>
  public class CaseRenamer
  {
    class RegexMatcher
    {
      public Regex OldValueRegex;
      public string NewValue;
    }
    class Renamer : ICurveRenamer
    {
      CaseRenamer m_renamer;
      public Renamer(CaseRenamer r)
      {
        m_renamer = r;
      }
      public string Rename(string original_name)
      {
        string new_name = original_name;
        string name = original_name;
        foreach (var r in m_renamer.m_rename_regex) {
          Match match = r.OldValueRegex.Match(name);
          if (match.Success) {
            new_name = r.OldValueRegex.Replace(name, r.NewValue);
            Console.WriteLine("Change name={0} to new_name={1}", name, new_name);
          }
        }
        return new_name;
      }
    }
    string m_input_path;
    List<RegexMatcher> m_rename_regex;
    Renamer m_renamer;
    public CaseRenamer(string rename_file, string input_path)
    {
      m_rename_regex = new List<RegexMatcher>();
      _ReadRenameDocument(rename_file);
      m_input_path = input_path;
      m_renamer = new Renamer(this);
    }

    void _ReadRenameDocument(string rename_file)
    {
      /*
       * Le fichier XML doit avoir le format suivant:
       *
       * <?xml version="1.0">
       * <root>
       *  <rename old="old_name" new="new_name" />
       * </root>
       *
       * Les expressions régulières sont autorisées. Elles utilisent le format
       * de la classe Regex.
       */
      Console.WriteLine("Reading rename file '{0}'", rename_file);
      XDocument doc = XDocument.Load(rename_file);
      foreach (var elem in doc.Root.Elements("rename")) {
        string old_v = (string)elem.Attribute("old");
        string new_v = (string)elem.Attribute("new");
        Console.WriteLine("Rename old={0} new={1}", old_v, new_v);
        RegexMatcher m = new RegexMatcher();
        m.OldValueRegex = new Regex(old_v);
        m.NewValue = new_v;
        m_rename_regex.Add(m);
      }
    }
    public void Execute()
    {
      ArcaneCaseReader.RenameCurves(m_input_path, m_renamer);
    }
  }
}
