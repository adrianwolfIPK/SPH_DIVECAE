import os
import ffmpeg
print("Imported from:", ffmpeg.__file__)
print("Module content:", dir(ffmpeg))

def pngs_to_mp4(directory, variable_name, output_file="output.mp4", fps=24):
    """
    Convert PNG frames named VARIABLENAME_0000.png, VARIABLENAME_0001.png, ...
    into an MP4 video.

    Args:
        directory (str): Path to folder containing PNGs.
        variable_name (str): Prefix before the frame number.
        output_file (str): Output MP4 filename.
        fps (int): Frames per second for the video.
    """

    # Example pattern: myimages/MyVar_0000.png
    ffmpeg_path = r"C:\Users\adr61871\Desktop\fisherman_test\ffmpeg\bin\ffmpeg.exe"

    input_pattern = os.path.join(directory, f"{variable_name}_%04d.png")
    output_pattern = os.path.join(directory, output_file)

    (
        ffmpeg
        .input(input_pattern, framerate=fps)
        .output(output_pattern, vcodec="libx264", pix_fmt="yuv420p")
        .run(cmd=ffmpeg_path)
    )

    print(f"Created video: {output_file}")

# Example usage:
# pngs_to_mp4("/path/to/images", "VARIABLENAME", "myvideo.mp4", fps=30)

path = r"Q:\IPK-Projekte\OE31000_Brooks_Digital-Twin\05_Arbeitsunterlagen\10_BRKS\DIVE\AX120_L"
pngs_to_mp4(path, r"animation-1", "ghul.mp4")