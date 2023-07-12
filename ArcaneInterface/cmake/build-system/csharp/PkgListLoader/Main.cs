using System;
using System.Text;
using System.Collections.Generic;
using System.IO;
using System.Xml;

namespace PkgListLoader
{
	class MainClass
	{
		static private void showHelp(Mono.Options.OptionSet opt_set)
        {
            Console.WriteLine("Options :");
            opt_set.WriteOptionDescriptions(Console.Out);
            Environment.Exit(0);
        }
		
	    public static string ToUpperCaseName(string s)
    	{
	        return s.Replace("-", "_").ToUpper();
    	}
		
		public static void Main(string[] args)
		{
			bool show_help = false;
			bool verbose = false;
			bool add_prefix = false;
			Mono.Options.OptionSet opt_set = new Mono.Options.OptionSet();
			opt_set.Add("h|help|?", "help message", v => { show_help = true; });
			opt_set.Add("v|verbose", "verbose mode", v => { verbose = true; });
			opt_set.Add("add_prefix", "add xml prefix", v => { add_prefix = true; });
			string[] remaining_args = opt_set.Parse(args).ToArray();
			 
			if (show_help || remaining_args.Length != 2) showHelp(opt_set);
			string pkglist_file = remaining_args[0];
			string output_file = remaining_args[1];
			
			if (verbose)
				Console.WriteLine("Loading {0} into {1}", pkglist_file, output_file);

      string prefix = "";
      if(add_prefix) {
 				prefix = "XML_";
 			}

			XmlDocument document = new XmlDocument();
			document.Load(pkglist_file);
		
			StringBuilder buffer = new StringBuilder();
			buffer.Append("#\n# Generated file -- DO NOT EDIT --\n#\n\n");
			
			XmlNodeList packages = document.SelectNodes(string.Format("//packages/package[@available='true']"));
			List<string> package_list = new List<string>();
			
			foreach (XmlNode package_node in packages)
      {
				XmlElement package = package_node as XmlElement;
				String name = package.GetAttribute("name");
				if (name.StartsWith("arcane_")) continue; // skips arcane libraries
				
				if (verbose)
					Console.WriteLine("Loading delegated package {0}...", name);
				String cmake_name = ToUpperCaseName(name);
				package_list.Add(cmake_name);
				
				buffer.AppendFormat("set({0}{1}_FOUND \"YES\")\n", prefix, cmake_name);
				buffer.AppendFormat("set({0}{1}_NAME \"{2}\")\n", prefix, cmake_name, name);
				XmlNodeList libraries = package.SelectNodes("lib-name");
				if(libraries.Count > 0) {
					buffer.AppendFormat("set({0}{1}_LIBRARIES ", prefix, cmake_name);
					foreach(XmlNode lib in libraries)
					{
						buffer.AppendFormat("\"{0}\" ", lib.InnerText);
					}
					buffer.Append(")\n");
				}
								
				XmlNodeList includes = package.SelectNodes("include");
				if(includes.Count > 0) {
					buffer.AppendFormat("set({0}{1}_INCLUDE_DIRS ", prefix, cmake_name);
					foreach(XmlNode inc in includes)
					{
						buffer.AppendFormat("\"{0}\" ", inc.InnerText);
					}
					buffer.Append(")\n");
				}

        XmlNodeList flags = package.SelectNodes("flags");
        if(flags.Count > 0) {
          buffer.AppendFormat("set({0}{1}_FLAGS ", prefix, cmake_name);
          foreach(XmlNode flag in flags)
          {
            buffer.AppendFormat("\"{0}\" ", flag.InnerText);
          }
          buffer.Append(")\n");
        }

        XmlNodeList target = package.SelectNodes("target-name");
        if(target.Count > 0) {
          buffer.AppendFormat("set({0}{1}_TARGET_NAME ", prefix, cmake_name);
          foreach(XmlNode t in target)
          {
            buffer.AppendFormat("\"{0}\" ", t.InnerText);
          }
          buffer.Append(")\n");
        }

			}	

			buffer.Append("set(DELEGATED_DEPENDENCIES ");
			buffer.Append(string.Join(" ", package_list.ToArray()));
			buffer.Append(")\n");
			
			File.WriteAllText(output_file, buffer.ToString());
		}
	}
}