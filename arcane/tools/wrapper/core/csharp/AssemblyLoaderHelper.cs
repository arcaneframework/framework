//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.Collections.Generic;
using System.Reflection;
using System.IO;

namespace Arcane
{
  /*!
    \brief Interface pour spécialiser le chargement  d'assembly.
    Cela est utilisé par les applications '.Net Core' qui ne
    peuvent pas utiliser Assembly.LoadFile().
  */
  public interface IAssemblyLoader
  {
    Assembly Load(string assembly_name);
  }

  public class NetCoreIndirectAssemblyLoader : IAssemblyLoader
  {
    MethodInfo m_get_default_method;
    MethodInfo m_load_assembly_from_path_method;
    public NetCoreIndirectAssemblyLoader(MethodInfo method1,MethodInfo method2)
    {
      m_get_default_method = method1;
      m_load_assembly_from_path_method = method2;
    }
    public Assembly Load(string name)
    {
      Debug.Write($"Loading assembly [.NetCore3 Indirect] '{name}'");
      object default_context =  m_get_default_method.Invoke(null, new object [0]);
      if (default_context==null)
        return null;
      object a = m_load_assembly_from_path_method.Invoke(default_context,new object[] { name });
      if (a != null)
        return a as Assembly;
      return null;
    }

    /*!
     * \brief Détecte si on peut utiliser le chargement via 'AssemblyLoadContext'.
     *
     * Pour cela, détecte si la méthode
     * 'System.Runtime.Loader.AssemblyLoadContext.Defaut.LoadFromAssemblyPath' pour
     * charger les assembly est disponible. Cela est nécessaire avec '.NetCore 3' car la
     * méthode 'Assembly.LoadPath' ne fonctionne pas car elle est spécifique au framework.
     */
    public static Arcane.IAssemblyLoader CheckCreateLoader()
    {
      Type type = Type.GetType("System.Runtime.Loader.AssemblyLoadContext");
      if (type==null){
        Debug.Write("No type 'AssemblyLoadContext' found");
        return null;
      }

      Debug.Write($"Has 'AssemblyLoadContext' type={type}");
      BindingFlags common_flags = BindingFlags.Public | BindingFlags.InvokeMethod | BindingFlags.FlattenHierarchy;
      string method_name = "get_Default";
      MethodInfo method = type.GetMethod(method_name, common_flags | BindingFlags.Static);
      if (method == null){
        Debug.Write(2,$"Can not find static method named '{method_name}' in type '{type}'");
        return null;
      }
      string method2_name = "LoadFromAssemblyPath";
      MethodInfo method2 = type.GetMethod(method2_name,common_flags | BindingFlags.Instance);
      if (method2 == null){
        Debug.Write(2,$"Can not find static method named '{method2_name}' in type '{type}'");
        return null;
      }

      Debug.Write(2,$"Found valid method '{method2_name}' in type '{type}'");
      return new NetCoreIndirectAssemblyLoader(method,method2);
    }
  }

  /*!
   * \brief Classe utilitaire pour charger une assembly et ses dépendances.
   */
  internal class AssemblyLoaderHelper
  {
    HashSet<string> m_already_loaded = new HashSet<string>();
    string m_main_assembly_location;

    internal static IAssemblyLoader Loader { get; set; }

    internal AssemblyLoaderHelper()
    {
      if (Loader==null)
        Loader = NetCoreIndirectAssemblyLoader.CheckCreateLoader();
    }

    Assembly _LoadAssembly(string full_path)
    {
      Debug.Write($"Trying to load assembly with path '{full_path}'");
      Assembly a = null;
      if (Loader!=null)
        a = Loader.Load(full_path);
      if (a == null)
        if (File.Exists(full_path))
          a = Assembly.LoadFile(full_path);
      return a;
    }

    /*!
     * \brief Charge l'assembly \a assembly_name.
     *
     * Si \a assembly_name n'est pas un chemin absoule, l'assembly doit
     * se trouver dans le répertoire courant.
     */
    internal Assembly LoadOneAssembly(string assembly_name)
    {
      string full_assembly_name = assembly_name;
      if (!Path.IsPathRooted(assembly_name))
        full_assembly_name = Path.Combine(Directory.GetCurrentDirectory(),assembly_name);
      Assembly a = _LoadAssembly(full_assembly_name);
      if (a == null)
        throw new ApplicationException($"Can not load assembly '{full_assembly_name}'");
      return a;
    }

    /*!
     * \brief Charge l'assembly \a assembly_name et toutes ces sous-assembly.
     *
     * Les sous assemblys doivent se trouver dans le même répertoire que
     * l'assembly principale.
     */
    internal void LoadSpecifiedAssembly(string assembly_name)
    {
      m_already_loaded.Clear();
      if (String.IsNullOrEmpty(assembly_name))
        return;
      Assembly a = LoadOneAssembly(assembly_name);
      if (a == null)
        throw new ApplicationException(String.Format("Can not load assembly '{0}'", assembly_name));
      m_main_assembly_location = Path.GetDirectoryName(a.Location);
      _LoadSubAssemblies(a);
    }

    void _LoadSubAssemblies(Assembly a)
    {
      if (a == null)
        return;
      m_already_loaded.Add(a.GetName().FullName);
      AssemblyName[] deps = a.GetReferencedAssemblies();
      if (deps == null)
        return;
      foreach (var sa in deps){
        _LoadSpecifiedAssembly(sa);
      }
    }
    /*!
     * \brief Tente de charger une assembly.
     *
     * Avec '.NetCore' il n'y a pas de résolution des chemins (sauf à utiliser
     * le package 'System.Runtime.Loader). Cela peut poser problème si une DLL chargée
     * utilise une autre DLL. Pour résoudre ce problème on cherche d'abord
     * pour une DLL si un fichier de même nom existe dans le répertoire de la DLL
     * de base.
     */
    void _LoadSpecifiedAssembly(AssemblyName aname)
    {
      if (aname==null)
        return;
      Debug.Write($"Analysing assembly '{aname}'");
      // Ne tente pas de lire les assembly qui sont signées
      // car normalement elles dépendent du framework.
      // (ce n'est pas obligatoirement le cas mais pour nous si)
      if (aname.GetPublicKeyToken().Length!=0)
        return;
      Debug.Write("Trying to load assembly '{0}'", aname);
      if (m_already_loaded.Contains(aname.FullName))
        return;
      string full_path = Path.Combine(m_main_assembly_location,aname.Name+".dll");

      // TODO: fusionner avec le code de LoadSpecificAssembly
      Assembly a = _LoadAssembly(full_path);
      if (a==null)
        a = Assembly.Load(aname);
      if (a == null)
        throw new ApplicationException($"Can not load assembly '{aname}'");
      _LoadSubAssemblies(a);
    }
  }
}
