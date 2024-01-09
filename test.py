import pkg_resources

dist = pkg_resources.get_distribution("prophet")
for dep in dist.requires():
    print(f"{dep.project_name} ({dep.specifier})")
    