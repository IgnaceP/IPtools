import gi
gi.require_version('Notify', '0.7')
from gi.repository import Notify, GdkPixbuf

def notifier(message):
    """
    Function to send a Linux notification
    """

    Notify.init("My Program Name")
    # Change application name

    # Create the notification object and show once
    notification = Notify.Notification.new('')
    notification.set_app_name("Python")
    notification.update(message)

    # Use GdkPixbuf to create the proper image type
    image = GdkPixbuf.Pixbuf.new_from_file("/home/ignace/Pictures/Logos/Python.png")

    # Use the GdkPixbuf image
    notification.set_icon_from_pixbuf(image)
    notification.set_image_from_pixbuf(image)

    notification.show()
