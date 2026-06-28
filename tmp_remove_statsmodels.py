import os, shutil, stat, subprocess, traceback

base = r"T:\OneDrive\1TB\School\PV_autoTs\.venv\Lib\site-packages"
paths = [
    os.path.join(base, "statsmodels"),
    os.path.join(base, "statsmodels-0.14.6.dist-info"),
]

def onerror(func, path, exc_info):
    try:
        os.chmod(path, stat.S_IWRITE)
    except Exception as e:
        print("chmod failed", path, e)
    try:
        if os.path.isdir(path) and not os.path.islink(path):
            os.rmdir(path)
        else:
            os.remove(path)
    except Exception as e:
        print("force delete failed for", path, e)


def try_rmdir_cmd(p):
    try:
        print('rmdir cmd:', p)
        subprocess.check_call(["cmd", "/c", "rmdir", "/s", "/q", p])
        print('rmdir cmd removed', p)
    except Exception as e:
        print('rmdir cmd failed', p, e)


for p in paths:
    try:
        print('\n--- checking:', p)
        print('exists:', os.path.exists(p))
        if os.path.exists(p):
            try:
                shutil.rmtree(p, onerror=onerror)
                print('shutil.rmtree removed', p)
            except Exception as e:
                print('shutil.rmtree failed for', p, e)
                try_rmdir_cmd(p)
            finally:
                if os.path.exists(p):
                    print('still exists after attempts:', p)
                else:
                    print('confirmed removed:', p)
        else:
            print('not present:', p)
    except Exception as e:
        print('exception while handling', p, e, traceback.format_exc())

print('\nRemaining statsmodels entries under site-packages:')
try:
    for x in os.listdir(base):
        if x.lower().startswith('statsmodels'):
            print(' -', x)
except Exception as e:
    print('listing failed', e)
print('\nDone')
