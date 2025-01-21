<?xml version="1.0" encoding="utf-8"?>
<!-- Created by Leo: https://leo-editor.github.io/leo-editor/leo_toc.html -->
<leo_file xmlns:leo="https://leo-editor.github.io/leo-editor/namespaces/leo-python-editor/1.1" >
<leo_header file_format="2"/>
<globals/>
<preferences/>
<find_panel_settings/>
<vnodes>
<v t="ekr.20250121070617.1"><vh>Readme</vh></v>
<v t="ekr.20250121054642.1" descendentVnodeUnknownAttributes="7d7100285803000000302e3071017d7102580b0000005f5f626f6f6b6d61726b7371037d7104580700000069735f6475706571054930300a73735805000000302e302e3271067d71075808000000616e6e6f7461746571087d71092858080000007072696f72697479710a4d0f27580a00000070726973657464617465710b580a000000323032312d30332d3330710c75735803000000302e31710d7d710e580b0000005f5f626f6f6b6d61726b73710f7d7110580700000069735f6475706571114930300a7373752e"><vh>Startup</vh>
<v t="ekr.20250121055447.2" descendentVnodeUnknownAttributes="7d71002858010000003071017d7102580b0000005f5f626f6f6b6d61726b7371037d7104580700000069735f6475706571054930300a73735803000000302e3271067d71075808000000616e6e6f7461746571087d71092858080000007072696f72697479710a4d0f27580a00000070726973657464617465710b580a000000323032312d30332d3330710c7573752e"><vh>@settings</vh>
<v t="ekr.20250121055447.163"><vh>@data history-list</vh></v>
<v t="ekr.20250121055447.154"><vh>@button backup</vh></v>
<v t="ekr.20250121055447.164" descendentVnodeUnknownAttributes="7d710058010000003071017d71025808000000616e6e6f7461746571037d71042858080000007072696f7269747971054d0f27580a000000707269736574646174657106580a000000323032312d30332d333071077573732e"><vh>@enabled-plugins</vh></v>
</v>
<v t="ekr.20250121055447.161" descendentVnodeUnknownAttributes="7d710058010000003071017d7102580b0000005f5f626f6f6b6d61726b7371037d7104580700000069735f6475706571054930300a7373732e"><vh>Scripts</vh>
<v t="ekr.20250121054955.1"><vh> Recursive import script</vh></v>
</v>
</v>
<v t="ekr.20250121055138.2"><vh>@file engine.py</vh></v>
<v t="ekr.20250121055138.19"><vh>@file nn.py</vh></v>
</vnodes>
<tnodes>
<t tx="ekr.20250121054642.1"></t>
<t tx="ekr.20250121054955.1">"""Recursively import all python files in a directory and clean the result."""
@language python
@tabwidth -4 # For a better match.
g.cls()
# dir_ = r'C:\Python\Python3.12\Lib\site-packages\micrograd'
dir_ = r'C:\Repos\manim'
c.recursiveImport(
    dir_=dir_,
    kind = '@clean', # '@auto', '@clean', '@nosent','@file',
    recursive = True,
    safe_at_file = True,
    theTypes = ['.py',],
    verbose = True,
)
if 1:
    last = c.lastTopLevel()
    last.expand()
    if last.hasChildren():
        last.firstChild().expand()
    c.redraw(last)
print('Done')</t>
<t tx="ekr.20250121055447.154">"""
Back up this .leo file.

os.environ['LEO_BACKUP'] must be the path to an existing (writable) directory.
"""
c.backup_helper(sub_dir='ekr-study')
</t>
<t tx="ekr.20250121055447.161" __bookmarks="7d7100580700000069735f6475706571014930300a732e"></t>
<t tx="ekr.20250121055447.163">
</t>
<t tx="ekr.20250121055447.164" annotate="7d71002858080000007072696f7269747971014d0f27580a000000707269736574646174657102580a000000323032312d30332d33307103752e">@language python

# Recommended plugins, from leoSettings.leo:

plugins_menu.py
# contextmenu.py      # Required by the vim.py and xemacs.py plugins.
mod_scripting.py
nav_qt.py
# Both VR plugins may be enabled.
viewrendered.py
# viewrendered3.py
</t>
<t tx="ekr.20250121055447.2" __bookmarks="7d7100580700000069735f6475706571014930300a732e">@language rest
@wrap

The @settings tree contains all active settings. 

Settings outside this tree have no effect.</t>
<t tx="ekr.20250121070617.1">@language rest

A study outline for micrograd: 
https://github.com/karpathy/micrograd

Video:
https://www.youtube.com/watch?v=VMj-3S1tku0&amp;list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ

@language python
</t>
</tnodes>
</leo_file>
